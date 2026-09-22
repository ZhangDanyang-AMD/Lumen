# MXFP4 DeepSeek V4 MoE 训练链路 — Design Spec

**Date:** 2026-09-21

**Status:** Draft

**Lifecycle:** Spec developing

**Scope:** gfx950 上 DeepSeek V4 routed-expert FC1/FC2 的 MXFP4 A4W4 训练链路；覆盖 Megatron expert-parallel 接线、前向、DGrad、WGrad、权重缓存、回退、测试与性能验收

**Primary repositories:** Lumen + AITER

**Review rule:** 用户审核前不得将状态改为 Approved

---

## 1. 摘要

当前 DSV4 的“FP8 MoE”链路已经能在 Megatron 的路由与通信外壳中运行，但计算核心并不是真正的 grouped FP8 training operator：

1. Megatron 完成 router、两次 permutation、EP all-to-all 和本地 expert sorting。
2. `TEGroupedMLP` 只作为上层 MoE 语义外壳，FC1/FC2 被替换成 Lumen grouped linear。
3. `LumenGroupedLinear.forward()` 在 Python 中按 expert 切片，并为每个非空 expert 单独调用 `quantized_linear()`。
4. 默认 FP8 recipe 是 blockwise、block size 128。Forward 和 DGrad 可走 FP8；WGrad 只有在单个 expert 的动态 token 数 `M_e` 可被 128 整除时才走 FP8，否则回退 BF16。
5. DSV4 在模型构建完成后通过环境变量开启 Lumen FP8，没有设置 Megatron `config.fp8`，因此 `TEGroupedMLP` 的 FP8 padding 和 router padding 默认都不会生效。

所以，当前链路应准确描述为：

> Megatron expert-sorted MoE + Lumen 逐 expert blockwise FP8 linear；它不是单次 grouped FP8 kernel，也不是端到端全 FP8 backward。

本设计选择以下目标方案：

> 保留 Megatron 的 router、dispatch/combine 和 BF16 数学语义，在 dispatch 后的 expert-sorted 边界引入 AITER grouped MXFP4 A4W4 training primitives，由 Lumen 负责 module/autograd、缓存、dispatch、fallback、checkpoint 和 optimizer 生命周期。

首版不直接给 AITER inference `fused_moe` 包 autograd。当前 AITER 的 DSV4 tuned rows 是 A8W4，而不是本设计的 A4W4；gfx950 A4W4 inference 路径也没有可直接复用的 EP training backward。

---

## 2. 证据与状态标记

本文使用四类标记，避免把实验能力写成已交付能力：

| 标记 | 含义 |
|---|---|
| **[Current]** | 当前 checkout 中已经存在并可由源码直接确认 |
| **[Reusable]** | 已在别的 Lumen/AITER 路径存在，但尚未接入 DSV4 grouped MoE |
| **[Design]** | 本 spec 冻结的实现要求 |
| **[Unverified]** | 需要 GPU、分布式或训练实验验证，当前不得作能力声明 |

源码基线：

| Repository | Branch | Commit |
|---|---|---|
| Lumen | `dev/mxfp4` | `9d85c8adb5159cc5765680bbc4c2230bb00e74e4` |
| AITER（runtime checkout） | `bench/ecfff3f-lumen` | `e35bb17f4f815903bf73598facedbb321e15af28` |
| Megatron-LM | `core_r0.15.0_rocm` | `1b754411c7777799fb17e260baa7145c0c6d71a2` |

Lumen HEAD 中 `third_party/aiter` gitlink 指向 `ecfff3fa80f906c5c421a35a7f5e52842f000559`，但当前 Python 实际导入 `/home/xdai/aiter/aiter/__init__.py`，其 checkout 为上表的 `e35bb17f4f815903bf73598facedbb321e15af28`。两者不是同一 revision。后续启动日志、测试报告和 benchmark 结果必须记录 `aiter.__file__`、runtime AITER commit、Lumen gitlink commit、GPU arch 和实际 kernel/config key；只记录 Lumen commit 或 submodule gitlink 不足以复现实验。

本次 Lumen 与 runtime AITER checkout 均包含未提交修改，Megatron-LM checkout 为 clean；因此本文证据对应 **dirty workspace snapshot**，不能只靠上述 commit 重建。任何用于验收的 run 必须归档三个仓库的 `git status --porcelain=v1`、tracked binary diff 的 SHA256，以及相关 untracked 文件清单与内容摘要；正式报告应附可恢复的 patch/artifact。

本文审计时的 snapshot fingerprint（不包含本文件，因为 `docs/` 被 ignore）：

| Repository | `git diff --binary` SHA256 | sorted untracked-content aggregate SHA256 |
|---|---|---|
| Lumen | `76720f664b318bfd1cbaec17f6f49debea949f3d6e1118089d978a1622a71202` | `f3f2c9554e67cbdd896d2864c27e90f0cfdbc3c11454b655fd7846b2bf63bf2d` |
| runtime AITER | `8f103472cb2af7a96bb27ce7db583397ae0e389627dd0a672bc279d94fc275a1` | `9f853d876724c2f1ae3811c0f3f0957358e8682d00e7129bc398c60832c826d1` |
| Megatron-LM | clean | none |

untracked aggregate 的计算口径是：按路径排序后逐文件计算 SHA256，再对完整的 `hash + path` 清单计算一次 SHA256。

行号基于 2026-09-21 的工作区快照，后续修改可能使其漂移。

---

## 3. 目标、非目标与成功条件

### 3.1 Goals

1. 为 DSV4 routed experts 的 FC1/FC2 提供完整 MXFP4 A4W4 forward、DGrad、WGrad 训练闭环。
2. 以 Megatron 已排序的本地 expert token 段为接口，不重写 router、top-k、EP all-to-all 或 combine。
3. 保留 BF16 model/checkpoint weight、现有 `weight0..weightN` 参数名和 distributed checkpoint 布局；optimizer precision 完全沿用对应 BF16 baseline，不因 MXFP4 改变。
4. 对 empty expert、ragged `M_e`、长尾路由分布、gradient accumulation、activation recompute 和 deferred WGrad 给出确定语义。
5. 新 GPU kernel 位于 AITER；Lumen 只负责集成、autograd、dispatch、缓存和可观测 fallback。
6. 基准必须调用真实生产 API，并验证实际 selected backend，不能用 PyTorch 模拟结果代替 kernel 性能。

### 3.2 Non-goals

以下内容不属于 v1：

- 量化 router、routing logits/probabilities、dispatch/combine 通信、shared expert、attention、embedding 或 lm-head。
- 修改 top-k=6、routing probability 作用位置、SwiGLU/clamp 数学语义。
- 直接训练化 AITER 的完整 inference `fused_moe`。
- 支持 expert tensor parallel，v1 固定 `ETP=1`。
- 支持 gfx942、gfx1250 或任意模型 shape；v1 production target 为 gfx950 上的 DSV4 shape。
- 把 A8W4 自动当作 A4W4 的 fallback，或按 shape 静默切换 A4W4/A8W4。
- 在没有 profile 证据前融合 router、permutation、weighted SwiGLU 与 FC2 activation quant，或融合通信。

### 3.3 成功条件

功能成功与性能成功分开判定：

- **P0 baseline-ready:** 顺序逐 expert MXFP4 路径通过外部数值语义、该路径自身的 cache lifecycle、checkpoint、EP smoke 和 fallback 可观测性测试。
- **P1 correctness-ready:** grouped 路径通过相同的外部数值/训练语义门槛、其自身的 grouped cache lifecycle 测试，并相对 P0 通过同 operands parity。
- **Production-ready:** P1 在 strict 模式下完成 DSV4 4-layer 与 43-layer 训练验收，所有 routed-expert FC1/FC2 的 fwd/dgrad/wgrad fallback counter 为 0，并达到第 17.6 节的性能门槛。
- **不成立的成功声明:** “能启动”“loss 有限”“某个 AITER inference shape 有 tuned row”均不足以证明 DSV4 MXFP4 MoE training 已完成。

---

## 4. 术语与固定精度定义

### 4.1 本文中的 MXFP4

本文的 `MXFP4` 固定指：

- activation operand: FP4 E2M1 packed，1×32 E8M0 scale；
- weight operand: FP4 E2M1 packed，32×32 E8M0 tile scale；
- GEMM recipe: **A4W4**；
- GEMM accumulator: FP32；forward/DGrad output storage: BF16；
- WGrad accumulator: FP32；普通 parameter grad storage 为 BF16，gradient accumulation fusion 写入目标 `main_grad` dtype，当前 DSV4 baseline 为 FP32；
- model/checkpoint parameter: BF16；optimizer precision 保持对应 BF16 baseline。

`A8W4` 指 activation 为 FP8 E4M3、weight 为 FP4 E2M1；外部资料有时称其为 `W4A8`，但本项目的日志、配置和报告只使用 canonical 名称 `A8W4`。它是不同 recipe，必须使用不同 capability、配置文件和统计项。

### 4.2 Shape 符号

对本地 expert `e`：

- `E_l`: 当前 rank 的 local expert 数；
- `M_e`: 当前 microbatch 中分配给 expert `e` 的 token 数；
- `M = sum(M_e)`；
- `K`: linear 输入宽度；
- `N`: linear 输出宽度；
- `P_e = ceil(M_e / 32) * 32`: WGrad 对 expert `e` 独立 padding 后的 token 维。

DSV4 v1 精确 shape：

| Stage | `K` | `N` | 每 expert weight |
|---|---:|---:|---|
| FC1 gate+up | 4096 | 4096 | `[4096, 4096]` |
| FC2 down projection | 2048 | 4096 | `[4096, 2048]` |

FC1 的 `N=4096` 来自 `2 × moe_ffn_hidden_size`；输出前半和后半仍按 Megatron 当前 `torch.chunk(..., 2)` 的 gate/up 顺序解释。

---

## 5. 当前 DSV4 FP8 MoE 链路审计

### 5.1 模型与并行拓扑

**[Current]**

`examples/dsv4/dsv4_megatron_args.sh:44-105` 固定：

- hidden size 4096；
- 256 个 routed experts；
- top-k 6；
- expert FFN hidden size 2048；
- SwiGLU；
- all-to-all token dispatcher；
- `--moe-grouped-gemm`。

默认并行：

| Profile | TP | PP | EP | ETP | Local experts/rank |
|---|---:|---:|---:|---:|---:|
| 4-layer | 8 | 1 | 8 | 1 | 32 |
| 43-layer Flash | 4 | 4 | 4 | 1 | 64 |

证据：

- 4-layer: `examples/dsv4/dsv4_megatron_args.sh:11-24`
- 43-layer: `examples/dsv4/dsv4_flash_mi300x_parallel.sh:2-14`
- expert/top-k/FFN: `examples/dsv4/dsv4_megatron_args.sh:75-86`

### 5.2 真实数据流

**[Current]**

```text
TopK router
  │
  ├─ local permutation
  ├─ EP all-to-all dispatch
  ├─ TP gather（如 dispatcher 配置需要）
  └─ local-expert sorting
       │
       ▼
  TEGroupedMLP 语义外壳
       │
       ├─ LumenColumnParallelGroupedLinear / FC1
       ├─ clamp + SwiGLU
       ├─ × routing probability
       └─ LumenRowParallelGroupedLinear / FC2
       │
       ▼
  local-expert unsort
  ├─ TP reduce-scatter（如适用）
  ├─ EP all-to-all combine
  └─ final unpermute
```

源码边界：

- DSV4 provider 在 `lumen/models/dsv4/megatron/spec_provider.py:26-35` 选择 `TEGroupedMLP`，并注入 Lumen grouped FC1/FC2。
- Megatron grouped MLP 在 `megatron/core/transformer/moe/experts.py:842-963` 执行 FC1、activation/probability 和 FC2。
- dispatch 的 local permutation、EP all-to-all、TP gather/local sort 位于 `token_dispatcher.py:552-704`。
- combine 的 local unsort、TP reduce-scatter、EP all-to-all 和 final unpermute 位于 `token_dispatcher.py:706-803`。

### 5.3 当前 grouped linear 实际不是 grouped kernel

**[Current]**

`lumen/modules/grouped_linear.py:126-167`：

1. 在 Python 中遍历 `num_gemms`；
2. 根据 `m_splits[i]` 切出 `x_i`；
3. 对每个非空 expert 单独调用一次 `quantized_linear()`；
4. 最后 `torch.cat(outputs)`。

该模块没有调用 `lumen/ops/gemm/grouped_gemm.py`。所以现状是“grouped module API + sequential per-expert GEMM”，不是 grouped execution。

`lumen/ops/gemm/grouped_gemm.py:166-336` 虽有 BF16、FP8 per-tensor/per-token/blockwise 和 MXFP8 forward dispatch，但没有 `mxfp4` 分支；其 `grouped_gemm_wgrad()` 在 quantized mode 下仍把每个 expert 转成 BF16 后顺序计算，见 `:386-440`。

### 5.4 FP8 启用路径

**[Current]**

```text
LUMEN_DSV4_LINEAR_FP8=1
  → dsv4_model_provider()
  → 模型构建完成
  → enable_fp8_for_dsv4_model()
  → enable_fp8_for_parallel_linear()
  → 每个 Lumen grouped module 设置 scaling_type/block_size
```

- 开关默认关闭：`lumen/models/dsv4/megatron/fp8.py:10-19`。
- 默认 recipe 为 `blockwise`，block size 128。
- 模型构建后启用：`lumen/models/dsv4/megatron/pretrain.py:141-160`。
- grouped modules 被纳入启用扫描：`lumen/models/dsv4/megatron/fp8.py:40-80`。

`LUMEN_DSV4_FP8_WGRAD` 虽被转换成 `linear_fp8_wgrad`（`lumen/models/dsv4/megatron/fp8.py:20-34`），但 `LumenGroupedLinear.forward()` 调用 `quantized_linear()` 时没有传入 `fp8_wgrad`（`lumen/modules/grouped_linear.py:126-154`）；functional API 的默认值又是 `True`（`lumen/ops/quantize/linear.py:3796-3808`）。此外，MXFP4 和 blockwise 专用 backward 分支也没有以该字段决定是否执行低精度 WGrad，见 `linear.py:3013-3030,3104-3122`。因此该环境变量目前不是 DSV4 grouped 路径的可靠行为开关。新 A4W4 contract 不继承这个失效开关；任何可配置 WGrad policy 都必须贯穿 module → op → autograd，并由 selected-kernel 测试证明实际生效。

DSV4 入口目前只在该环境变量打开时安装 `fp8_param_gather_hook`，见：

- `examples/dsv4/pretrain_dsv4_megatron.py:48-53`
- `examples/dsv4/finetune_dsv4_megatron.py:56-61`

它没有安装 `mxfp4_weight_cache_hook`。通用 `--linear-fp4` CLI 已存在于 `lumen/patches/builders/megatron_args.py:261-327`，但 DSV4 的 `add_dsv4_pretrain_args()` 当前只应用 `dsv4` tags，见 `lumen/patches/builders/dsv4.py:100-105`。`register_dsv4_megatron_cli()` 虽存在于 `lumen/models/dsv4/megatron/fp8.py:84-88`，当前 DSV4 entrypoint 未调用。

### 5.5 当前 FP8 forward/backward 精度行为

**[Current]**

默认 blockwise FP8 下，每个非空 expert 独立执行：

| Pass | 当前行为 | 对 DSV4 shape 的结果 |
|---|---|---|
| Forward | activation/weight blockwise FP8，A8W8 GEMM，BF16 output | `K/N` 均为 128 对齐，可走 FP8 |
| DGrad | `dY` 量化；1D blockwise weight scale 不能直接转置，因此优先从 BF16 model weight 重新量化 `W^T` | `N/K` 对齐时可走 FP8 |
| WGrad | `dY^T @ X` 的 token 轴要求 `M_e % 128 == 0` | 不对齐的 expert 回退 BF16 |

关键代码为 `lumen/ops/quantize/linear.py:3091-3235`：

- DGrad 对齐条件只要求 `N`、`K` 可被 block size 整除，见 `:3117-3122`。
- WGrad 要求动态 `M` 可被 block size 整除，见 `:3120-3122`。
- blockwise DGrad 从 BF16 model weight 构造转置量化 operand，见 `:3172-3193`。
- WGrad 若没有 columnwise quantized operands 或 kernel 拒绝，则回退 BF16，见 `:3215-3235`。

**[Unverified inference]** expert token 数由 top-k 路由动态决定，通常不会恰好全部满足 `M_e % 128 == 0`。因此当前 DSV4 默认 FP8 路径的 WGrad 预计大量使用 BF16；必须由新增 backend counter 和真实路由直方图确认，不能仅凭静态推断报告具体比例。

### 5.6 Padding 断点

**[Current]**

Megatron 的 grouped MLP padding 仅在 `config.fp8` 为真时构造并执行：

- 初始化：`megatron/core/transformer/moe/experts.py:817-820`
- forward padding/unpadding：`:859-868,956-958`

router padding 只由 `config.moe_router_padding_for_fp8` 控制：

- `megatron/core/transformer/moe/token_dispatcher.py:577-583`

当前 DSV4 是模型构建后给 Lumen modules 设置 quantization 状态，并未设置 `TransformerConfig.fp8`，脚本也未开启 router padding。因此不能把 Megatron 的 FP8 padding 当作当前 WGrad 对齐保证。

MXFP4 v1 不依赖 router padding。它必须在 WGrad 内对每个 expert segment 独立 pad 到 32，并在输出处按原 segment 截断。

### 5.7 Scaling state 与 cache 风险

**[Current]**

- `_enable_quantization_for_parallel_linear()` 为一个 grouped module 创建一个 `ScalingManager`，见 `lumen/models/megatron.py:486-530`。
- grouped loop 调用 `quantized_linear()` 时未传 `tensor_id`，因此所有 experts 使用默认 `"weight"`，见 `grouped_linear.py:138-152` 和 `linear.py:3796-3808`。
- 对默认 blockwise recipe，该 collision 不直接影响 amax history；若改为 delayed scaling，则不同 experts 会共享同一个 history key，语义错误。
- native dense/parallel MXFP4 路径在 `_do_gemm()` 中使用 per-step weight cache，见 `lumen/modules/parallel_linear.py:334-375`。
- grouped module 直接调用 `quantized_linear()`，绕过 `_do_gemm()`；若仅把 `scaling_type` 改成 `mxfp4`，会对每个 expert、每个 microbatch 重复量化并构建转置 weight。

### 5.8 当前测试缺口

**[Current]**

- `tests/ops/test_grouped_gemm.py:9-15` 声明的覆盖主要是 BF16 forward/WGrad 和 FP8 forward。
- 没有 grouped MXFP4、grouped DGrad、完整 module backward、cache、EP integration 测试。
- `tests/modules/test_grouped_linear.py:45-69` 仍访问 `m.weights/m.biases`，而实现已改为注册 `weight0..weightN`/`bias0..biasN`（`lumen/modules/grouped_linear.py:99-124`）；这表明现有 construction test 与 production API 已漂移，不能作为当前参数布局已覆盖的证据。
- `tests/modules/test_grouped_linear.py:225-239` 的量化测试只检查 enable 状态。
- `tests/modules/test_grouped_linear.py:245-272` 的 benchmark test 只断言 elapsed time 大于 0，不能作为性能验收。

### 5.9 当前 recompute 与 overlap 约束

**[Current]**

DSV4 4-layer、43-layer 和 profile 启动脚本默认使用 full activation recompute，例如 `examples/dsv4/run_dsv4_flash_pretrain_inner.sh:63-69`；这些脚本当前也没有打开 EP A2A overlap 或 shared-expert overlap。Megatron 在 `overlap_moe_expert_parallel_comm` 打开时明确要求 `recompute_granularity != 'full'` 且 recompute method/num-layers 均为空，见 `megatron/core/transformer/transformer_config.py:1448-1477`。

因此 v1/P1 不把 EP A2A overlap 作为默认能力或成功条件。它只能在 P2 中以独立配置、显存变化、数值回归和端到端 profile 重新验收，不能为追求 overlap 静默改变当前训练的 recompute 策略。

---

## 6. 可复用能力与不可外推结论

### 6.1 Lumen dense MXFP4 training

**[Reusable]**

现有 dense `quantized_linear(..., scaling_type="mxfp4")` 已提供：

- activation 1×32 E8M0 scale、RTN；
- weight 32×32 E8M0 tile scale、RTN；
- packed FP4 E2M1；
- A4W4 forward，BF16 output；
- gradient stochastic rounding；
- DGrad 使用 forward weight 的 packed transpose；
- WGrad 使用 dual-layout quant 和 deterministic H16/RHT，`g=16`；
- ragged `M` 在 backward 内 pad 到 32，并裁剪 dX；
- 32-bit operand indexing 的 dispatch 前 fail-closed；
- 每 optimizer step 的 weight cache 与 Parameter `_version` 校验。

主要证据：

- MXFP4 dispatch: `lumen/ops/quantize/linear.py:2162-2166`
- int32 guard: `:2179-2216,2599-2609`
- forward 与 `W^T` operand: `:2219-2331`
- ragged M、dual-layout gradient、DGrad/WGrad: `:2334-2560`
- deterministic H16/RHT: `:74-89`
- 32×32 weight scale 的转置一致性：`lumen/ops/quantize/ops.py:619-645`
- weight cache: `lumen/quantize/__init__.py:543-653`
- optimizer invalidation: `lumen/quantize/__init__.py:1176-1245`

这些能力证明 A4W4 训练 recipe 的单 GEMM 数值路径可用，但不证明 grouped MoE 的性能、segment isolation、EP integration 或端到端收敛。

### 6.2 AITER 现有 MoE MXFP4

**[Reusable]**

AITER 已有 gfx950 FlyDSL A4W4 inference forward：

- `aiter/ops/flydsl/test_flydsl_moe_a4w4.py:4-18` 覆盖 stage1、stage2 和 E2E forward；
- activation 与 weight 都是 FP4，见 `:41-43`；
- stage1 是 fused gate+up GEMM + activation，见 `aiter/ops/flydsl/moe_kernels.py:1099-1152`；
- stage2 是 down projection + routed reduction，见 `:1453-1505`。

这些 kernel/layout/quantization 经验可以复用，但其 API 包含 inference routing/sorting/fusion 语义，不是 Megatron 已完成 A2A/local-sort 后需要的裸 grouped linear。

### 6.3 当前 DSV4 tuned config 是 A8W4

**[Current]**

`aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv:122-137` 的 DSV4 rows：

- hidden 4096；
- intermediate 2048；
- experts 256；
- top-k 6；
- activation dtype `torch.float8_e4m3fn`；
- weight dtype `torch.float4_e2m1fn_x2`。

因此它们是 **A8W4**，不是本 spec 的 **A4W4**。另外，config lookup key 包含 expert 数，见 `aiter/fused_moe.py:1770-1783`；runtime 从本地 weight bank 得到 `E`，见 `:569-570`。EP4/EP8 的 local `E=64/32` 不会命中现有 `E=256` rows。

### 6.4 AITER 当前不存在本 spec 所需 backward

**[Current]**

- `fused_moe` / `fused_moe_2stages` 是 forward API，没有配套 autograd、MXFP4 DGrad 或 MXFP4 WGrad。
- AITER 存在独立 Triton `moe_wgrad`，见 `aiter/ops/triton/moe/moe_wgrad.py:47-211`，但它接收普通 grad/input，不消费 MXFP4 operands/scales，也未接入 `fused_moe` backward。
- gfx950 A4W4 inference 的 `output_aux` sort 在有 `expert_mask` 时明确抛出 `NotImplementedError`，见 `aiter/fused_moe.py:702-713`。
- gfx1250 grouped A4W4/A8W4 测试走完整 public `fused_moe` forward，见 `op_tests/test_flydsl_grouped_gemm_gfx1250.py:6-22`；它不能作为 gfx950 DSV4 training 支持证据。

---

## 7. 方案比较

| 方案 | 描述 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A. 顺序复用 dense MXFP4 | 沿当前 Python expert loop，对每个 `X_e/W_e` 调 dense MXFP4 autograd | 最小改动；最快建立数值、cache、checkpoint baseline；ragged M 已支持 | 32/64 experts × FC1/FC2 × fwd/dgrad/wgrad，launch 与 Python 开销大；不是 grouped kernel | **选为 P0 correctness baseline，不作为性能完成** |
| B. 直接给 AITER `fused_moe` 包 autograd | 把 router/sort/stage1/stage2/reduce 全部纳入一个 autograd op | 理论上融合最多 | 与当前 Megatron dispatch 边界冲突；改变概率与 activation 语义风险高；gfx950 A4W4 EP受限；无 backward；现有 DSV4 tuned rows 是 A8W4 | **拒绝作为 v1** |
| C. Expert-sorted grouped linear training API | 在 Megatron dispatch 后对 FC1/FC2 分别做 grouped A4W4 fwd/dgrad/wgrad | 保留上层语义与 checkpoint；API 边界清晰；可独立测试和调优；兼容 EP local-E | 需要新 AITER grouped training kernels 和 Lumen custom autograd | **选为 P1 目标方案；production 需另过 promotion gate** |

选择 C 的核心理由是：grouped GEMM 是需要加速的最小稳定边界，router、通信和概率加权已经由 Megatron 正确表达，不需要为获取 kernel 融合而重写整条 MoE 数学图。

---

## 8. 决策表

| ID | 决策 | 冻结选择 | 理由 |
|---|---|---|---|
| D1 | 精度 recipe | MXFP4 A4W4 | 与 dense training 能力一致；不混入 A8W4 |
| D2 | 集成边界 | dispatch 后的 expert-sorted grouped FC1/FC2 | 保持 Megatron router/EP/combine |
| D3 | v1 硬件 | gfx950 | 当前 A4W4 kernel 与目标平台 |
| D4 | 并行范围 | EP4/EP8，ETP=1 | 覆盖 DSV4 43-layer/4-layer 默认拓扑 |
| D5 | 参数/optimizer 状态 | BF16 model/checkpoint weight；FP32 main param/main grad；moment dtype 沿用 profile baseline | 保留训练、optimizer 与 checkpoint 语义 |
| D6 | Weight scale | canonical 32×32 E8M0 tile grid | 保证 forward weight 与 DGrad transpose 一致 |
| D7 | Activation scale | 1×32 E8M0 | 沿用 dense A4W4 recipe |
| D8 | Rounding | weight/activation RTN，gradient SR | 沿用已验证 dense recipe |
| D9 | WGrad padding | 每 expert 独立 pad32，H16/RHT 每段重启 | 防止 expert 间数据污染 |
| D10 | 上层 activation | 保留 BF16 clamp + SwiGLU + routing probability | 不改变模型数学语义 |
| D11 | 权重缓存 | 每 optimizer step 构造 fwd + DGrad transpose cache | 避免每 microbatch 重复量化/转置 |
| D12 | Fallback | grouped MXFP4 → sequential MXFP4 → BF16 grouped | correctness 可恢复且路径可观测 |
| D13 | 用户开关 | 接入标准 `--linear-fp4`，拒绝与 FP8 同开 | 不再增加另一个环境变量格式开关 |
| D14 | Kernel ownership | AITER | 遵守 Lumen/AITER 边界 |
| D15 | P0/P1 定义 | P0 顺序正确性 baseline；P1 grouped operator；production 单独 promotion | 防止将正确性 fallback 当性能结果 |

---

## 9. 目标架构与数据流

### 9.1 端到端边界

**[Design]**

```text
BF16 hidden states
  → Megatron router/top-k (BF16/FP32 as today)
  → local permutation
  → EP all-to-all
  → local expert sorting
  → X_sorted [sum M_e, 4096], group_sizes[E_l]
       │
       ├─ grouped MXFP4 FC1 A4W4
       │    W1 cache [E_l, 4096, 4096]
       │    → BF16 gate+up [sum M_e, 4096]
       │
       ├─ existing BF16 clamp + SwiGLU
       ├─ existing BF16 × routing probability
       │
       └─ grouped MXFP4 FC2 A4W4
            W2 cache [E_l, 4096, 2048]
            → BF16 expert output [sum M_e, 4096]
  → local unsort
  → TP reduce-scatter（如适用）
  → EP all-to-all combine
  → final unpermute
```

以下边界保持不变：

- `TEGroupedMLP` 对 FC1/activation/probability/FC2 的调用顺序；
- `activation_func_clamp_value=10` 的 clamp；
- probability 在 SwiGLU 输出后相乘；
- all-to-all dispatcher 与 checkpoint key。

### 9.2 Lumen module contract

`LumenGroupedLinear` 的对外签名保持兼容：

```python
forward(
    x: Tensor,                 # [sum(M_e), K], BF16
    m_splits: Sequence[int],   # length E_l
    m_splits_gpu: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]
```

要求：

1. `sum(m_splits) == x.shape[0]`，所有值非负。
2. `m_splits_gpu` 若提供，必须是 CUDA contiguous `int32` 且与 host counts 一致。
3. 当前 `TEGroupedMLP` 把 counts 转成 Python list。v1 在 `m_splits_gpu is None` 时复用预分配 buffer，把 host list 拷回 CUDA；不得为此再触发一次 D2H 同步。
4. P1 不改变返回值、bias 语义或 checkpoint layout。
5. DSV4 v1 的 bias 被 `--disable-bias-linear` 禁用；若收到 bias，auto 模式走 BF16，strict 模式构建时拒绝。

### 9.3 Lumen autograd contract

新增 grouped autograd glue 的逻辑签名：

```python
GroupedMXFP4LinearFunction.apply(
    x,
    group_sizes_gpu,
    module_metadata,
    *expert_weights,
)
```

Forward：

1. 校验 shape、dtype、arch 和 local-E。
2. 获取或构建 grouped weight cache。
3. 对 expert-sorted `x` 做 segment-aware activation quant。
4. 调用 AITER grouped forward。
5. 保存 compact MXFP4 operands、group offsets 和 cache generation；不保存额外 BF16 weight bank。

Backward：

1. 对 `dY` 做 segment-aware dual-layout quant，gradient 使用 SR。
2. grouped DGrad 使用 cached `W^T` operand。
3. grouped WGrad 使用每 expert 独立 pad32 的 `dY_e^T/X_e^T`。
4. 普通 autograd 返回与 `weight0..weightN` 一一对应、dtype 与 BF16 model parameter 相同的 gradients。
5. gradient accumulation fusion 模式直接写入各 expert 的目标 `main_grad` buffer；当前 DSV4 baseline 要求支持 FP32，且相应 autograd 返回 `None`。
6. deferred WGrad 模式在 backward 时完成量化并捕获 packed operands；延后阶段只执行 GEMM，避免调度顺序改变 stochastic rounding。

### 9.4 AITER public API

**[Design]**

AITER 提供 expert-sorted training primitives，不接受 global router logits、top-k ids、top-k weights 或 expert mask。v1 public contract 固定使用 canonical contiguous storage；任何 kernel-private swizzle/preshuffle 都由 AITER wrapper 从这些字段派生并缓存，不得重新量化 BF16 source。

```python
from dataclasses import dataclass
from typing import Literal, Sequence
from torch import Tensor


@dataclass(frozen=True)
class GroupedMXFP4WeightOperands:
    data: Tensor          # uint8 CUDA contiguous [E_l, N, K/2], packed E2M1
    scale: Tensor         # uint8 CUDA contiguous [E_l, N/32, K/32], E8M0
    data_t: Tensor        # uint8 CUDA contiguous [E_l, K, N/2], packed E2M1
    scale_t: Tensor       # uint8 CUDA contiguous [E_l, K/32, N/32], E8M0
    layout_id: Literal["mxfp4_weight_tile32x32_v1"]


@dataclass(frozen=True)
class GroupedMXFP4ActivationOperands:
    row_data: Tensor      # uint8 CUDA contiguous [M, D/2], packed E2M1
    row_scale: Tensor     # uint8 CUDA contiguous [M, D/32], E8M0
    transposed_data: Tensor
    # uint8 CUDA contiguous [sum_e(D*P_e/2)]; expert e view [D, P_e/2]
    transposed_scale: Tensor
    # uint8 CUDA contiguous [sum_e(D*P_e/32)]; expert e view [D, P_e/32], E8M0
    group_offsets: Tensor # int32 CUDA [E_l+1], prefix sum of M_e
    padded_offsets: Tensor
    # int32 CUDA [E_l+1], prefix sum of P_e in token units
    feature_dim: int      # D = K for X, D = N for dY
    rounding: Literal["rtn", "sr"]
    rht_id: Literal["hadamard16_all_plus_v1"]
    philox_seed: int | None
    row_philox_offset: int | None
    transposed_philox_offset: int | None
    next_philox_offset: int | None
    layout_id: Literal["mxfp4_segmented_row_and_t_v1"]


grouped_mxfp4_weight_quant_2d(
    weights: Sequence[Tensor], # exactly E_l BF16 CUDA contiguous [N,K]
) -> GroupedMXFP4WeightOperands

grouped_mxfp4_quant_forward_activation(
    x_bf16: Tensor,            # BF16 CUDA contiguous [M,K]
    group_sizes_gpu: Tensor,   # int32 CUDA contiguous [E_l]
) -> GroupedMXFP4ActivationOperands

grouped_mxfp4_quant_backward_gradient(
    dy_bf16: Tensor,           # BF16 CUDA contiguous [M,N]
    group_sizes_gpu: Tensor,   # int32 CUDA contiguous [E_l]
    *,
    philox_seed: int,
    philox_offset: int,        # offset unit: one 32-bit random draw
) -> GroupedMXFP4ActivationOperands

grouped_mxfp4_linear_fwd(
    x_operand: GroupedMXFP4ActivationOperands,
    weight_operand: GroupedMXFP4WeightOperands,
    *,
    config_id: str | None = None,
) -> Tensor                   # BF16 [M,N]

grouped_mxfp4_linear_dgrad(
    dy_operand: GroupedMXFP4ActivationOperands,
    weight_operand: GroupedMXFP4WeightOperands,
    *,
    config_id: str | None = None,
) -> Tensor                   # BF16 [M,K]

grouped_mxfp4_linear_wgrad(
    dy_operand: GroupedMXFP4ActivationOperands,
    x_operand: GroupedMXFP4ActivationOperands,
    *,
    out: Sequence[Tensor],     # exactly E_l contiguous [N,K], all BF16 or all FP32
    accumulate: bool,
    config_id: str | None = None,
) -> None
```

固定语义：

1. `block_size=32`、weight/forward-activation RTN、gradient SR 和两个 `layout_id` 都是 v1 ABI，不作为可变 keyword。
2. `grouped_mxfp4_quant_forward_activation()` 的 `row_*` 是不做 RHT 的 1×32 RTN operand；其 `transposed_*` 是逐 expert pad32、逐段重启 H16 后的 RTN WGrad operand。
3. `grouped_mxfp4_quant_backward_gradient()` 的 `row_*` 是不做 RHT 的 1×32 SR DGrad operand；其 `transposed_*` 是使用同一 H16 约定的逐段 SR WGrad operand。
4. `X` 与 `dY` 的 `group_offsets`、`padded_offsets`、`rht_id` 必须完全相同，WGrad wrapper 在 launch 前验证。
5. `config_id=None` 表示按第 13.1 节完整 key 查表；传入字符串表示测试/benchmark 强制指定一个已注册 config。两者都必须在 telemetry 中解析成实际 config ID。
6. WGrad 只使用 mandatory `out` buffers，不返回二义性的 stacked/sequence variant。普通 autograd 分配 BF16 per-parameter buffers；gradient accumulation fusion 直接传入 FP32 `main_grad` buffers。`accumulate=False` 覆盖写，`True` 原位累加。

API 不持有 `Parameter`、optimizer 或 Python module 状态。AITER wrapper 负责：

- 参数验证；
- device pointer table；
- kernel dispatch；
- layout tag；
- kernel repr/trace name；
- tuned config lookup；
- 同步暴露真实 backend 名。

Lumen 负责 cache 生命周期、Parameter 映射、autograd、fallback 和训练框架接线。

### 9.5 为什么不传 router 数据

进入 grouped FC1 前，Megatron 已经完成：

- top-k expansion；
- EP dispatch；
- local-expert sorting。

因此 operator 只需要 `x_sorted + group_sizes`。这样：

- 不依赖 AITER inference `expert_mask`；
- 不重复 sorting；
- 不改变 probability gradient；
- local-E 自然为 32 或 64；
- 可独立验证每个 linear pass。

---

## 10. 数值与布局契约

### 10.1 Weight cache

对 expert `e`：

| 名称 | Shape | 含义 |
|---|---|---|
| `W_e` | `[N, K]` BF16 | model/checkpoint weight |
| `Wq_e` | `[N, K/2]` packed FP4 | forward B operand |
| `Sw_e` | `[N/32, K/32]` E8M0 | canonical 32×32 tile scale |
| `Wtq_e` | `[K, N/2]` packed FP4 | DGrad B operand |
| `Swt_e` | `[K/32, N/32]` E8M0 | `Sw_e.T` 的逻辑布局 |

Grouped cache 的逻辑 shape：

- `Wq`: `[E_l, N, K/2]`
- `Sw`: `[E_l, N/32, K/32]`
- `Wtq`: `[E_l, K, N/2]`
- `Swt`: `[E_l, K/32, N/32]`

如果 kernel 需要 per-row 展开的 scale 或特殊 swizzle，它必须从 canonical 32×32 grid 派生；不得重新以 1×32 规则量化 BF16 weight。否则 forward 的 `Q(W)` 与 DGrad 使用的 weight 不再是同一量化算子的转置，破坏链式法则一致性。

### 10.2 Forward

对每个 expert：

```text
X_e [M_e,K] BF16
  --RTN, 1×32-->
Xq_e [M_e,K/2], Sx_e [M_e,K/32]

Y_e = GEMM_A4W4(Xq_e, Wq_e, Sx_e, Sw_e)
  --FP32 accumulate, cast-->
Y_e [M_e,N] BF16 storage
```

Forward 不要求 `M_e` 对齐到 32；每行独立 1×32 quant，kernel 尾块必须 masked。

### 10.3 DGrad

```text
dY_e [M_e,N] BF16
  --SR, 1×32-->
dYq_e

dX_e = GEMM_A4W4(dYq_e, Wtq_e)
  --FP32 accumulate, cast-->
dX_e [M_e,K] BF16 storage
```

要求：

- 使用 forward cache 中对应的 `Wtq_e/Swt_e`；
- 不允许从更新后的 BF16 model weight 临时再量化；
- ragged `M_e` 不得使 DGrad 静默切换 BF16；
- kernel 不支持时按第 12 节回退，并计数。

### 10.4 WGrad

```text
dW_e = dY_e^T @ X_e

P_e = ceil(M_e / 32) * 32
dY_e^T: [N,P_e] --H16 + SR--> dYtq_e, Sdy_t [N,P_e/32]
X_e^T : [K,P_e] --H16 + RTN--> Xtq_e,  Sx_t  [K,P_e/32]
dW_e  : [N,K] --FP32 accumulate--> target grad buffer
```

强制不变量：

1. 每个 expert 独立 pad，不能把 expert `e` 的尾部和 expert `e+1` 的开头放进同一 32-value quant block。
2. deterministic H16/RHT 的 `g=16` 在每个 expert segment 起点重启；同一 expert 的 `dY_e^T` 与 `X_e^T` 必须使用相同的 `hadamard16_all_plus_v1` sign/order/segment origin，不能独立生成 transform。
3. `dY` 使用 stochastic rounding，`X` 使用 round-to-nearest。
4. padding 值为 0，不参与 scale amax，不贡献 dW。
5. `M_e=0` 不启动 0-M kernel；`accumulate=False` 时该 expert 输出全零，`accumulate=True` 时保持目标 buffer 不变。
6. `P_e`、group offsets 和 padded offsets 必须由 quant 返回的同一份 metadata 提供，WGrad 不得自行重算。
7. WGrad 内部使用 FP32 accumulator；普通 autograd 的目标 buffer 为 BF16，gradient accumulation fusion 的 `main_grad` 目标必须支持 FP32。最终存储 dtype 由 mandatory `out` buffer 决定。

### 10.5 Stochastic rounding 与可复现性

只有 gradient quant 使用 SR。`grouped_mxfp4_quant_backward_gradient()` 的 RNG contract 固定为：

- `philox_offset` 的单位是一份 32-bit random draw；
- row-major `dY` stream 从 `row_philox_offset = philox_offset` 开始，消费 `M × N` 个 draw；
- transposed H16 `dY^T` stream 从 `transposed_philox_offset = philox_offset + M × N` 开始，按 expert 顺序消费 `sum(P_e × N)` 个 draw，包含 padding 位置；
- 返回 `next_philox_offset = philox_offset + M × N + sum(P_e × N)`；
- random draw 到逻辑 `(expert, row, column, layout)` 的映射不得依赖 kernel tile、wave 调度或 expert 是否合并到同一次 launch；
- 相同 seed、offset、shape 和 `group_sizes` 必须 bitwise 重放 packed gradient operands；row 与 transposed stream 不得重叠。

deferred WGrad 在原 backward 时生成并保存 packed operands 及上述 metadata，延后执行不得再次消费 RNG 或重新量化。

### 10.6 Routing probability 与 activation

P1 不融合 activation：

```text
FC1 BF16 output
  → clamp(gate≤10, linear∈[-10,10])
  → SiLU(gate) × linear
  → × permuted_probs
  → BF16
  → FC2 activation quant
```

该顺序匹配 `megatron/core/transformer/moe/experts.py:924-940`。routing probability 的梯度继续由 PyTorch/Megatron autograd 计算。

### 10.7 32-bit indexing

每个 grouped API 在 launch 前验证：

- 每个 operand 的元素数；
- packed byte offset；
- scale offset；
- pointer table offset；
- `sum(P_e)`。

任一 index 可能超过 signed 32-bit 时：

- strict 模式：在 launch 前抛出包含 layer/role/shape 的异常；
- auto 模式：选择 sequential MXFP4 或 BF16；
- 禁止“先 launch，再捕获 illegal memory access”，因为 HIP context 可能已经损坏。

---

## 11. Cache 与 optimizer 生命周期

### 11.1 保留参数、optimizer 与 checkpoint

**[Design]**

- `weight0..weightN` 继续是独立 BF16 `Parameter`。
- 不把 checkpoint 主表示改成单一 `[E,N,K]` Parameter。
- precision-aware optimizer 的 main parameter 默认保持 FP32，见 `megatron/training/arguments.py:3781-3784` 和 `megatron/core/optimizer/optimizer_config.py:72`。
- DSV4 训练继续使用 `--accumulate-allreduce-grads-in-fp32`（4-layer `run_dsv4_4layer_pretrain_inner.sh:82`；43-layer `run_dsv4_flash_pretrain_inner.sh:113`）；gradient accumulation fusion 因而必须能直接累加到 FP32 `main_grad`。
- Adam moments 不统一改 dtype：43-layer profile 当前在 `run_dsv4_flash_pretrain_inner.sh:22-23,121-122` 显式默认 BF16，4-layer 未覆盖参数时沿用 Megatron FP32 默认。每个 run 必须记录 main-param、main-grad、exp-avg 与 exp-avg-sq dtype。
- grouped packed cache 是 non-persistent runtime state，不进入 `state_dict`。
- 现有 `LumenGroupedLinear._sharded_state_dict_grouped()` 的 global expert 映射保持不变。

### 11.2 Cache key

一个 grouped weight cache entry 至少绑定：

```text
(
  tuple(source_parameter_identity),
  tuple(source_parameter_version),
  E_l, N, K,
  device, source_dtype,
  gfx_arch, cu_count,
  fc_role,                    # fc1 / fc2
  quant_recipe,               # mxfp4-a4w4
  canonical_scale_layout,     # tile32x32
  packed_data_layout,
  kernel_layout_signature,
)
```

`Parameter._version` 是 correctness 的最终兜底。optimizer hook 只负责尽早释放旧 buffer，正确性不得只依赖 hook 是否被安装。

### 11.3 构造策略

- **P0:** 每个 expert 复用 dense `_mxfp4_cached_weight`，不构造永久 BF16 stack。
- **P1:** AITER grouped quant 接受独立 source tensors 的 pointer table，直接写连续 packed cache。
- 允许在 P1 bring-up 期间使用一次性 BF16 stack 作为受控 correctness fallback，但该路径必须有独立 counter，不能进入 production benchmark 或 release 配置。

缓存内容：

- forward packed weights/scales；
- DGrad packed transposed weights/scales；
- layout metadata；
- source versions；
- backend/config identity。

### 11.4 Invalidation

- BF16 weight 任一 `_version` 改变时，下一次 forward 必须 cache miss。
- optimizer 成功更新后 eager invalidate。
- overflow/skipped step 不得推进 cache generation；即使保守 invalidate，也不能把旧 version 标成新 version。
- checkpoint load、device move、dtype change、module re-shard 后必须 invalidate。
- activation recompute 在同一个 optimizer step 内复用相同 weight cache。

---

## 12. Dispatch、回退与可观测性

### 12.1 Backend 顺序

**[Design]**

```text
1. AITER grouped MXFP4 training kernel
2. Lumen sequential per-expert dense MXFP4
3. AITER/Lumen BF16 grouped GEMM
```

三条路径必须具有相同 shape、bias、empty-expert、autograd 和 checkpoint 语义。

### 12.2 Capability preflight

模型构建阶段检查：

- device arch 为 gfx950；
- AITER public grouped MXFP4 training API 可导入；
- runtime `aiter.__file__` 可解析到预期 checkout，runtime commit 与允许列表一致；
- `K`、`N` 满足 32 对齐；
- `ETP=1`；
- local-E 为 32 或 64；
- DSV4 bias 关闭；
- `--linear-fp4` 未与 FP8 开关同时启用；
- strict 模式下存在对应 FC1/FC2 kernel/config。

strict/release 模式若 runtime AITER import path 或 commit 不在声明的允许集合中，必须在模型构建前失败；开发模式至少输出一次显式 warning，不能只在 benchmark 结束后才暴露依赖漂移。

运行阶段检查：

- `group_sizes` 长度与 local-E 相同；
- 所有 count 非负且 sum 等于输入行数；
- host/GPU counts 一致；
- operand layout tag 与 kernel 要求一致；
- index 范围安全。

### 12.3 用户配置

格式选择使用现有标准开关：

```text
--linear-fp4
```

新增的 MoE 参数只控制 backend 和失败策略，不重新定义格式：

```text
--lumen-moe-mxfp4-backend {auto,grouped,sequential}
--lumen-moe-mxfp4-strict
```

语义：

- `auto`: grouped 不可用时按 fallback chain 降级；
- `grouped`: 请求 grouped；非 strict 时仍允许记录后降级；
- `sequential`: P0/reference 路径；
- `strict`: 任意 routed-expert FC1/FC2 非预期降级立即失败。

DSV4 v1 中 `--linear-fp4` 的受支持 scope 是 routed expert FC1/FC2。启动日志必须打印 `scope=routed_experts` 和实际 module/expert-bank 数；router、shared expert、attention、embedding、lm-head 保持 BF16。未来扩展到 all-linears 需要独立验收，不由该开关在 DSV4 中静默扩大范围。

兼容要求：

- 保留 `LUMEN_DSV4_LINEAR_FP8`；
- 若环境 FP8 与 `--linear-fp4` 同时设置，启动阶段报错；
- 不增加 `LUMEN_DSV4_LINEAR_MXFP4`；
- DSV4 parser 必须接入 common Megatron quant args；
- DSV4 training setup 必须安装 MXFP4 cache invalidation hook。

### 12.4 Fallback telemetry

每次选择路径时可查询：

- requested recipe；
- selected backend；
- Lumen/Megatron commit、各仓库 dirty flag/patch digest、`aiter.__file__`、runtime AITER commit、Lumen AITER gitlink commit，以及 GPU arch/CU 数；
- kernel/config name 与完整 tuning key；
- layer name 与 FC1/FC2 role；
- `E_l, M, N, K`；
- non-empty expert 数、`max(M_e)` 和 token bucket；
- fallback reason enum；
- grouped/sequential/BF16 累计次数。

日志要求：

- 首次发生某个 `(layer, role, reason)` 时 warning；
- 后续只增加 counter，避免日志洪泛；
- fallback counters 在所有 TP/PP/EP/DP ranks 上收集并求和，provenance 通过 all-gather 收集；rank 0 只负责展示全局结果，不能只读取本 rank；
- 所有 rank 的 runtime AITER commit 与 import root 必须一致；GPU arch/CU 和 selected kernel/config 可按 rank 分组展示，但不得省略非零 rank；
- strict run 中任一 rank 出现 fallback 即判失败：本 rank 不继续用 fallback 结果训练，并在下一安全的分布式同步边界传播 failure flag，使所有 ranks 一致退出，避免单 rank 异常造成 collective hang；
- 只有 backend 首次探测/warmup 或显式 correctness/debug validation 才同步暴露异步 kernel 错误，再允许 `try_backends()` 进入下一候选；
- 成功 backend 按完整 op/kernel key 缓存后，steady-state 与 timed path 不增加设备同步；
- CUDA/HIP graph capture 前必须预热并冻结所有会用到的 backend key，capture 内禁止 fallback probing、JIT/config lookup 或 host-side warning；
- BF16 fallback 不能记录成 MXFP4 success。

建议 reason enum：

```text
UNSUPPORTED_ARCH
UNSUPPORTED_SHAPE
UNSUPPORTED_LOCAL_EXPERTS
MISSING_AITER_API
MISSING_TUNED_CONFIG
INVALID_GROUP_SIZES
INDEX_OVERFLOW
KERNEL_REJECTED
KERNEL_RUNTIME_ERROR
BIAS_UNSUPPORTED
FORCED_SEQUENTIAL
FORCED_BF16
```

---

## 13. AITER tuning 与 kernel representation

### 13.1 Tuning key

P1 config key 至少包含：

```text
gfx, cu_count, pass, fc_role,
E_l, M_total_bucket, nonempty_experts_bucket, max_Me_bucket,
N, K,
activation_layout, weight_layout, scale_layout,
output_dtype, accumulate
```

原因：同一个 `M_total` 在均匀和长尾路由下有不同的 occupancy/尾块行为；只用 global E=256 或总 token 数无法代表 EP4/EP8 的真实 local workload。

必须覆盖：

- `E_l=32`：4-layer EP8；
- `E_l=64`：43-layer EP4；
- FC1 `(N=4096,K=4096)`；
- FC2 `(N=4096,K=2048)`；
- fwd、dgrad、wgrad；
- cache cold/hot；
- balanced、long-tail、many-empty buckets。

### 13.2 Config ownership

- 唯一持久化表为 AITER 的 `aiter/configs/model_configs/dsv4_a4w4_train_grouped_gfx950.csv`，只允许写入已验证的 tuned rows；不能复用 `dsv4_fp8fp4_tuned_fmoe.csv`。
- 未命中表时由 `aiter/ops/flydsl/grouped_mxfp4_linear.py` 的版本化 `heuristic_v1` 选择默认 config，并记录 `config_source=heuristic, config_id=heuristic_v1:<resolved fields>`；heuristic 不写入 CSV，也不能表述为 tuned。
- tuned row 的晋升流程固定为：对第 13.1 节完整 key 运行 AITER public-wrapper benchmark → 通过第 16.1 节 correctness matrix → 保存 raw result/provenance digest → code review 后写入 CSV。每行记录完整 key、config fields、benchmark artifact digest 和生成该行的 AITER commit。
- strict production run 要求命中 CSV 中的 exact tuned row；heuristic 只允许用于非 strict bring-up/调优。
- Lumen 不携带 kernel tuning CSV；

### 13.3 Kernel repr

Profiler 中至少区分：

```text
grouped_mxfp4_weight_quant_2d
grouped_mxfp4_activation_quant_fwd
grouped_mxfp4_activation_quant_wgrad
grouped_mxfp4_linear_fwd_fc1
grouped_mxfp4_linear_fwd_fc2
grouped_mxfp4_linear_dgrad_fc1
grouped_mxfp4_linear_dgrad_fc2
grouped_mxfp4_linear_wgrad_fc1
grouped_mxfp4_linear_wgrad_fc2
```

repr 中应带 `E_l/M_bucket/N/K/tile/config`，便于从 trace 判断真实路径。

---

## 14. Repository ownership 与文件变更

### 14.1 AITER

| 文件 | 变更 |
|---|---|
| `aiter/ops/grouped_mxfp4_linear.py` | 新 public wrapper、参数验证、capability 与 metadata |
| `aiter/ops/flydsl/grouped_mxfp4_linear.py` | gfx950 launcher/dispatch |
| `aiter/ops/flydsl/kernels/grouped_mxfp4_quant.py` | canonical weight quant、segment-aware row/transpose quant |
| `aiter/ops/flydsl/kernels/grouped_mxfp4_gemm.py` | grouped fwd/dgrad/wgrad kernels |
| `aiter/__init__.py` | public API export |
| `aiter/configs/model_configs/dsv4_a4w4_train_grouped_gfx950.csv` | local-E 32/64 的已验证 tuned rows；不存 heuristic rows |
| `op_tests/test_grouped_mxfp4_linear.py` | kernel correctness |
| `op_tests/op_benchmarks/flydsl/bench_grouped_mxfp4_linear.py` | kernel benchmark |

AITER patch 必须遵循其 kernel 提交流程，包含 standalone test、benchmark、tuning config 和可识别 repr。

### 14.2 Lumen

| 文件 | 变更 |
|---|---|
| `lumen/models/dsv4/megatron/quantization.py`（新） | 统一 BF16/FP8/MXFP4 选择、冲突检查、启动摘要 |
| `lumen/models/dsv4/megatron/pretrain.py` | 模型构建后应用标准 MXFP4 配置 |
| `lumen/patches/builders/dsv4.py` | 接入 common quant CLI 与 MoE backend/strict 参数 |
| `lumen/patches/training/megatron_hooks.py` | DSV4 安装 grouped MXFP4 cache invalidation |
| `lumen/modules/grouped_linear.py` | unique tensor IDs、P0 sequential、P1 grouped dispatch、cache/autograd |
| `lumen/ops/gemm/grouped_gemm.py` | MXFP4 adapter、backend result 与 fallback chain |
| `lumen/ops/dispatch.py` | AITER grouped MXFP4 capability probe |
| `lumen/quantize/__init__.py` | grouped multi-weight cache/version/invalidation helper |
| `tests/ops/test_grouped_gemm.py` | MXFP4 fwd/dgrad/wgrad 与 fallback |
| `tests/modules/test_grouped_linear.py` | module/autograd/cache/checkpoint |
| `tests/models/dsv4/test_mxfp4_moe_training.py` | CLI、provider、EP integration 与 training contract |
| `benchmarks/bench_dsv4_mxfp4_moe.py`（新） | production API microbenchmark |

不在 Lumen 中新增 Triton/FlyDSL/HIP kernel。

---

## 15. 实现阶段

### P0 — 可验证的顺序 MXFP4 correctness baseline

目的：先证明 DSV4 grouped module 的训练语义与 dense A4W4 一致，不做性能承诺。

交付：

1. DSV4 接入标准 `--linear-fp4`，并与环境 FP8 互斥。
2. 仅对 routed-expert grouped FC1/FC2 启用 `mxfp4`、block size 32。
3. 每个 expert 使用唯一 tensor ID。
4. 每个 `weightN` 使用 per-step cache 和 `_version` 校验。
5. 顺序逐 expert fwd/dgrad/wgrad 支持 empty/ragged `M_e`。
6. 安装 optimizer invalidation。
7. 完成 4-layer EP8 smoke、checkpoint round trip 和 backend telemetry。
8. 保持当前 full recompute 配置，不顺带开启 EP/shared-expert overlap。

P0 日志必须明确显示 `backend=sequential_mxfp4`。不得用“grouped MXFP4 kernel”描述该阶段。

### P1 — Grouped training operator

交付：

1. AITER public expert-sorted grouped weight quant、forward、DGrad、WGrad API。
2. gfx950 local-E 32/64、DSV4 FC1/FC2、真实 token buckets 的 tuning。
3. Lumen grouped autograd 与 pointer-table weight/cache 接线。
4. empty expert、ragged per-expert pad32、gradient accumulation、deferred WGrad。
5. strict fallback 与 backend counters。
6. 4-layer EP8 correctness 验收；43-layer EP4 留作 production promotion。
7. 保持当前 full recompute 基线；overlap 不计入 P1 收益。

P1 可以在满足 correctness-merge gate 后以 experimental 状态合入；只有继续满足第 17.6 节和第 18 节的 production promotion gate，才可默认启用或称为 production-ready。

### P2 — 仅由 profile 驱动的融合

候选项：

- weighted SwiGLU + FC2 input quant；
- dispatch/compute overlap；
- delayed WGrad overlap；
- counts GPU-resident fast path；
- A8W4 实验 recipe。

P2 的前提是保持 router/probability gradients、activation clamp、checkpoint 和 P1 数值门槛；未达到前不得并入 P1。

---

## 16. 测试计划

所有 FP4/GPU 数值测试必须在 CUDA/ROCm device 上运行；gradient reference weight 必须是 leaf tensor。

### 16.1 AITER kernel tests

使用两级、彼此独立的 oracle，均不得调用被测 grouped wrapper：

1. **Kernel parity oracle:** 固定 packed bytes、E8M0 scales、offsets 和 Philox metadata，将同一 operands 反量化为 FP32，逐 expert 用 FP32 matmul，再 cast 到目标 output dtype。它隔离 kernel/layout/indexing 错误，不重复计算量化误差。
2. **Recipe accuracy oracle:** 从原始 BF16 `X/W/dY` 做独立 BF16 per-expert 数学实现，用于衡量 A4W4 recipe 本身的误差和训练 SNR。

矩阵：

- `E_l ∈ {32,64}`；
- `M_e ∈ {0,1,15,16,31,32,33,63,64,127,128,129,257}`；
- balanced、single-hot、Zipf/long-tail、half-empty、all-empty；
- FC1 `(4096,4096)` 与 FC2 `(4096,2048)`；
- fwd、dgrad、wgrad；
- `accumulate=False/True`；
- WGrad BF16 parameter-grad 与 FP32 `main_grad` output buffers；
- cold/hot layout；
- deterministic seed/offset bitwise 重放与 `next_philox_offset` 校验。

专门的 segment isolation case：

```text
M_0=31, M_1=1, M_2=33
```

用 sentinel 值验证 expert 0 的 pad 区、expert 1 的唯一 token 和 expert 2 的首 block 互不共享 scale/RHT segment。

断言：

- shape/dtype；
- finite；
- empty expert dW 为零；
- `sum(group_sizes)==M`；
- backend/config name；
- 无越界或跨 expert 污染；
- `X/dY` 的 `group_offsets`、`padded_offsets` 与 `rht_id` 完全一致；
- 对 kernel parity oracle，fwd/dgrad/wgrad SNR 均 ≥ 30 dB，且至少 99% 元素满足 `atol=0.5, rtol=0.02`；
- 对 BF16 recipe oracle，fwd SNR ≥ 12 dB、dX SNR ≥ 12 dB、dW SNR ≥ 10 dB；
- 在相同 quantized operands/seed 下，grouped 相对 sequential MXFP4 的 SNR 下降不超过 0.5 dB。

上述绝对 floor 来自当前 dense MXFP4 的保守测试门槛；release 前还必须在真实 DSV4 activation/gradient 分布上确认，不得因 synthetic test 通过就宣称收敛等价。

### 16.2 Lumen op dispatch tests

- capability probe 成功/失败；
- runtime AITER import path/commit allowlist 与 strict/release fail-closed；
- 强制 grouped、sequential、BF16；
- AITER import 缺失；
- kernel 同步错误；
- unsupported arch/shape/local-E；
- int32 overflow；
- strict 模式拒绝 fallback；
- auto 模式 fallback 顺序与 reason counter；
- multi-rank counter reduce/provenance gather，且任一 rank fallback 使 strict run 全局失败；
- backend warmup 后 steady-state 无同步，graph capture 内不发生 probing/JIT/fallback；
- selected backend 断言，防止 BF16 假成功。

### 16.3 Lumen module/autograd tests

- 先修复遗留的 `m.weights/m.biases` 断言，使 construction test 对齐 `weight0..weightN`/`bias0..biasN`；
- `weight0..weightN` 参数与 state_dict key 不变；
- BF16 model grad 与 FP32 `main_grad` 两种 WGrad destination dtype；
- MXFP4 不改变 main-param/main-grad/moment dtype 或 optimizer checkpoint schema；
- FC1/FC2 forward、dX、每 expert dW；
- routing probability gradient；
- empty experts；
- ragged `M_e`；
- unique tensor ID；
- cache cold miss、同 step hit、weight `_version` 变化 miss；
- optimizer successful step invalidation；
- skipped/overflow step 不产生错误 generation；
- activation recompute；
- gradient accumulation fusion；
- deferred WGrad；
- WGrad policy/config 必须改变实际 selected kernel 或明确拒绝，不只改变 Namespace 值；
- bias 非空时的 strict/error 与 auto fallback；
- checkpoint save/load 后首个 forward 重建 cache。

### 16.4 Megatron integration tests

1. 单卡 EP1 小 shape，验证 API 和 autograd。
2. 8 GPU、4-layer、TP8/EP8/ETP1。
3. 16 GPU、43-layer、TP4/PP4/EP4/ETP1。
4. 对每层验证 `tokens_per_expert` 与 output segment 保持一致。
5. 验证 dispatch/combine、router 和 shared expert 仍为原路径。
6. 验证 strict 模式所有 routed expert passes 的 fallback counter 为 0。
7. 验证日志中的 Lumen/Megatron/runtime AITER commit、gitlink、import path、GPU arch 与实际进程一致。

### 16.5 训练测试

固定：

- 同一初始 checkpoint；
- 同一数据顺序与 seed；
- 同一 LR、warmup、optimizer、clip、loss scale；
- 同一 main-param、main-grad、exp-avg 和 exp-avg-sq dtype；
- 同一 TP/PP/EP/ETP；
- 不允许为获得 green run 修改已经与 BF16 对齐的训练参数。

阶段：

- 4-layer smoke：至少 20 optimizer steps，无 NaN/Inf、deadlock、非法访存。
- 4-layer paired run：至少 100 optimizer steps，BF16 与 MXFP4 同配置。
- 43-layer distributed smoke：至少 20 optimizer steps。
- production convergence run：4-layer 至少 500 steps；43-layer 至少 100 steps。

收敛门槛：

- 所有 optimizer step 的 loss 和 grad norm 有限；
- MXFP4 最终 validation loss 与 BF16 的差值不超过 `max(2 × BF16 seed-to-seed 标准差, BF16 loss 的 2%)`；
- overflow/skip rate 相对 BF16 增加不超过 1 个百分点；
- 不允许使用 BF16 fallback counter 非零的 run 作为“full MXFP4”收敛结果。

---

## 17. Benchmark 计划

### 17.1 原则

1. AITER kernel microbenchmark 调用 AITER public wrapper；Lumen integration/full-step benchmark 调用真实 Lumen production module/API。两类都必须断言 selected backend，不用纯 PyTorch 循环冒充 production kernel。
2. Lumen GPU 计时统一使用 `benchmarks.bench_utils.cuda_timer(..., trim_pct=10)`；包含 collective 的 case 固定 `dist_barrier=True`。
3. 多卡进程组固定使用 `backend="cpu:gloo,cuda:nccl"` 和当前 rank 的 CUDA `device_id`，避免 host collective 与 GPU collective backend 歧义。
4. kernel/local-block case 至少 20 次 warmup、100 次测量；从两端各裁剪 10% 样本后报告 trimmed mean、median、p95、standard deviation 和 CV。完整训练 step 使用第 17.4 节的 wall-clock 协议。
5. 比较顺序采用 A/B/A 或 ABBA，减少温度、频率和 cache 漂移。
6. 固定 Lumen/Megatron/runtime AITER commit、dirty patch digest、容器、GPU arch/CU、seed、batch、sequence length、optimizer 和并行拓扑。
7. 每个结果附 `aiter.__file__`、Lumen AITER gitlink、selected backend、kernel repr/config key 和 fallback counters。
8. 若 overlap 实验需要关闭 full recompute，必须作为独立配置报告显存和吞吐；不得把 recompute 策略变化带来的收益归因于 grouped MXFP4 kernel。

### 17.2 Workload

从 BF16 DSV4 run 捕获真实 `tokens_per_expert`，生成：

- p10/p50/p90/p99 total-token buckets；
- balanced；
- long-tail；
- many-empty；
- single-hot worst case。

不能只用均匀 synthetic counts。

### 17.3 分项计时

分别报告：

- weight cache cold build；
- weight cache hot hit；
- activation quant；
- FC1 forward；
- FC2 forward；
- FC1/FC2 DGrad；
- FC1/FC2 WGrad；
- full local expert block；
- dispatch/combine；
- full training step。

`full local expert block` 的计时边界固定为 hot-cache lookup → FC1 activation quant/GEMM → 原 BF16 clamp/SwiGLU/probability multiply → FC2 activation quant/GEMM；包含 hot-cache lookup 和两次 activation quant，不包含 router、dispatch/combine、optimizer step 或 cold weight-cache build。P0/P1/current-FP8 必须使用同一边界。

若评估 overlap，必须同时报告：

- compute-only；
- comm-only；
- sequential compute+comm；
- overlap；
- hidden time (`hidden_ms`)；
- speedup；
- overlap ratio/efficiency。

### 17.4 Full-step throughput 协议

CUDA event microbenchmark 不能作为完整训练 step 的硬门槛，因为 full step 还包含 host 调度、collective 和 CPU optimizer offload。4-layer 与 43-layer 的主吞吐指标固定采用 steady-state wall-clock window：

1. 先完成 backend/JIT/cache 预热和 20 个不计时 optimizer steps；计时窗口内禁止首次编译、fallback probing、checkpoint、evaluation 或额外 profiling。
2. 每个 profile 测量至少 100 个连续、未 overflow/skip 的 optimizer steps；若发生 skip，该窗口作废并单独报告原因，不能从样本中静默删除。
3. window 起点在 batch 已就绪之后；终点在最后一个 optimizer update、CPU offload 工作和相关 CUDA work 全部完成之后。窗口起止各执行一次 device synchronize 和 world barrier，窗口内部不注入额外 barrier/synchronize。
4. 每个 rank 使用 `time.perf_counter_ns()` 记录相同 train-step 边界的原始逐 step delta；窗口结束后一次性 all-gather。主 wall time 取各 rank 整个窗口 elapsed 的最大值，诊断用逐 step latency 取相同 step index 上各 rank delta 的最大值。
5. 主吞吐 `tokens/s = measured non-padding global tokens / max-rank window seconds`；同时报告 `samples/s = successful_steps × global_batch_size / max-rank window seconds`。第 17.6 节 full-step gate 使用 `tokens/s`。
6. 保留所有 rank 的 raw per-step samples 和 window totals；95% interval 使用保持时间顺序的 block bootstrap（block length 5 steps），不能把相关 step 当成独立同分布样本。
7. BF16、当前 FP8、P0、P1 使用相同数据、GBS/MBS、sequence length、recompute、optimizer/offload、日志频率和测量窗口；任一配置差异都使该对比无效。

### 17.5 Baselines

同一 API/shape 下比较：

1. BF16 grouped GEMM；
2. 当前 DSV4 sequential blockwise FP8，保留其真实 BF16 fallback 行为并报告各 pass counter；
3. P0 sequential per-expert MXFP4；
4. P1 grouped MXFP4；
5. 可选 A8W4 inference 数据只能列为旁证，不进入 training speedup 主表。

### 17.6 性能门槛

P1 correctness merge 不以未经测量的理论 speedup 为前提。标记 production-ready 前必须满足：

- 在真实 token histogram 加权后的 local expert block 上，P1 相对 P0 至少 1.20×；
- 4-layer full-step throughput 相对 P0 至少 1.10×；
- 43-layer full-step throughput 相对 BF16 至少 1.05×；
- 4-layer 与 43-layer full-step throughput 相对当前 blockwise FP8 均不得回退（speedup ratio ≥ 1.00×）；
- routing-distribution p90 workload 的 median local-block latency 不得比 P0 慢超过 5%；
- 对所有 speedup 下限，95% bootstrap confidence interval 的 lower bound 必须达到阈值；对 p90 workload 的 latency ratio `P1/P0`，95% interval 的 upper bound 必须 ≤ 1.05；
- cold cache cost 单独报告，不得混入 hot-path 数字后平均隐藏。

若未达门槛，功能可保留为 experimental，但不得默认启用或宣传为 production speedup。

---

## 18. 验收清单

### P0 exit criteria

- [ ] DSV4 parser 接受 `--linear-fp4`，并拒绝与 FP8 同开。
- [ ] 启动摘要明确 `recipe=a4w4, scope=routed_experts, backend=sequential_mxfp4`。
- [ ] 每个 expert 使用唯一 tensor ID。
- [ ] 同 optimizer step 内 weight cache 命中，更新后失效。
- [ ] FC1/FC2 exact shape 的 ragged fwd/dgrad/wgrad 通过。
- [ ] 4-layer EP8 20-step smoke 通过。
- [ ] checkpoint round trip 参数 key 无变化。
- [ ] 所有 fallback 可计数；结果不得标为 grouped performance。

### P1 correctness-merge criteria

- [ ] AITER grouped weight quant、fwd、dgrad、wgrad API 和 tests 合入。
- [ ] local-E 32/64、FC1/FC2、empty/long-tail/ragged matrix 全通过。
- [ ] Lumen module/autograd/cache/optimizer/deferred WGrad tests 全通过。
- [ ] 4-layer EP8 strict smoke 中 fallback counter 为 0。
- [ ] kernel parity、recipe SNR 与 4-layer paired 100-step training 门槛通过。
- [ ] A4W4 与 A8W4 的日志、config 和报告名称无混用。

### Production promotion criteria

- [ ] 43-layer EP4 strict smoke 中 fallback counter 为 0。
- [ ] 4-layer 500-step 与 43-layer 100-step convergence 门槛通过。
- [ ] benchmark 使用真实 API、真实路由 bucket，并通过性能门槛。
- [ ] profiler 能区分每个 MXFP4 pass 和 config。
- [ ] benchmark artifact 包含完整 runtime provenance、dirty patch digest、raw samples 和 bootstrap interval。

### Spec approval criteria

- [ ] Lumen 与 AITER owner 同意 API/ownership 边界。
- [ ] 训练 owner 同意精度 recipe、scope 和 convergence gate。
- [ ] benchmark owner 同意 workload、计时和性能 gate。
- [ ] 无未决占位符，且 shape/layout 均已定义。
- [ ] 用户审核后才把状态从 Draft 改为 Approved。

---

## 19. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 小且不均匀的 `M_e` 使 grouped kernel 利用率低 | P1 不比 sequential 快 | 以真实 histogram 调优；key 包含 nonempty/max-M buckets |
| 32×32 canonical weight scale 与现有 inference 1×32 layout 不同 | 直接复用 kernel 会读错或改变数值 | AITER 从 canonical grid 派生 layout，禁止重新量化 |
| expert segment padding 混合 | WGrad 数值污染，难以从 loss 定位 | 独立 offsets/helper + sentinel isolation test |
| `dY^T/X^T` 使用不同 H16 transform | WGrad 不再等价于共同正交变换后的乘积 | 固定相同 `rht_id`/segment origin，launch 前校验 metadata |
| `weight0..N` 分散存储导致 staging 开销 | 额外 BF16 显存和 copy | P1 使用 pointer-table quant；stack 只准 P0/bring-up 并计数 |
| cache 未失效 | 长期使用 step-0 weight，loss 悄然停滞 | `_version` correctness key + optimizer eager invalidation |
| fallback 隐藏 kernel 缺陷 | 看似成功但实际 BF16 | strict mode、backend assertion、结束汇总 |
| 只汇总 rank 0 或单 rank 先抛错 | 漏报 fallback，或其他 rank 卡在 collective | all-rank reduce/gather；在安全边界一致失败 |
| steady-state 每次同步探测错误 | 性能失真并破坏 graph capture | 只在 warmup/debug 同步；缓存 backend，capture 前预热 |
| deferred WGrad 改变 SR 随机序列 | 可复现性和收敛漂移 | 显式 Philox ranges；backward 时完成 quant，closure 只保留 GEMM |
| WGrad 把 optimizer destination 错误固定为 BF16 | FP32 `main_grad` 写入错误或额外 cast/copy | mandatory typed `out` buffers；按 BF16 baseline 保持 optimizer dtype |
| host list → GPU counts | 小 shape 下吞吐损失 | 复用 buffer并单独计时；P2 再做 GPU-resident 接口 |
| AITER tuned key 使用 global E | EP4/EP8 未命中 | config 固定 local-E 32/64 |
| 完整 fused inference API 改变模型数学语义 | router/prob gradient 错误 | v1 只接 expert-sorted grouped linear |

---

## 20. 后续但不阻塞 P1 的问题

以下均采用“默认不做”的冻结策略，不阻塞当前实现：

- 是否将 weighted SwiGLU 与 FC2 activation quant 融合：默认不融合，只有 profile 证明收益且梯度测试通过后进入 P2。
- 是否增加 A8W4 training recipe：默认不增加；需要独立 spec、CLI 名称和收敛验收。
- 是否把 counts 全程保留在 GPU：v1 允许从已存在的 host list 拷回预分配 CUDA buffer；P2 再评估 Megatron 接口扩展。
- 是否支持 ETP>1：v1 fail closed；后续需重新定义 local weight shard、scale grid 和 collective。
- 是否扩展到 shared expert：v1 保持 BF16；后续走 dense MXFP4 的独立验收。

---

## 21. Out of Scope

- Router/top-k kernel 重写。
- MoE auxiliary loss 或 load-balancing loss 的数值变更。
- EP dispatch/combine 的量化通信。
- shared expert、attention、embedding、lm-head MXFP4。
- FP4 参数持久化或 FP4 optimizer state。
- inference checkpoint 导出。
- gfx942/gfx1250 production support。
- ETP>1。
- 无验证的 1.6× 或其他端到端性能承诺。

---

## 22. 关键源码索引

| 主题 | 位置 |
|---|---|
| DSV4 模型/专家配置 | `examples/dsv4/dsv4_megatron_args.sh:44-105` |
| 4-layer parallel | `examples/dsv4/dsv4_megatron_args.sh:11-24` |
| 43-layer parallel | `examples/dsv4/dsv4_flash_mi300x_parallel.sh:2-14` |
| Grouped MLP provider | `lumen/models/dsv4/megatron/spec_provider.py:26-35` |
| Megatron grouped FC1/act/FC2 | `megatron/core/transformer/moe/experts.py:746-963` |
| Token dispatch/sort/combine | `megatron/core/transformer/moe/token_dispatcher.py:552-803` |
| DSV4 FP8 post-build enable | `lumen/models/dsv4/megatron/fp8.py:10-80` |
| DSV4 provider enable point | `lumen/models/dsv4/megatron/pretrain.py:141-160` |
| DSV4 full recompute default | `examples/dsv4/run_dsv4_flash_pretrain_inner.sh:63-69` |
| Full recompute/EP overlap exclusion | `megatron/core/transformer/transformer_config.py:1448-1477` |
| Megatron FP32 main-param default | `megatron/training/arguments.py:3781-3784`; `megatron/core/optimizer/optimizer_config.py:72` |
| DSV4 FP32 grad/BF16 moment profile | `examples/dsv4/run_dsv4_flash_pretrain_inner.sh:22-23,113,121-122`; `examples/dsv4/run_dsv4_4layer_pretrain_inner.sh:82` |
| Sequential grouped module | `lumen/modules/grouped_linear.py:126-167` |
| Grouped test parameter API drift | `tests/modules/test_grouped_linear.py:45-69`; `lumen/modules/grouped_linear.py:99-124` |
| Existing grouped GEMM modes | `lumen/ops/gemm/grouped_gemm.py:166-440` |
| Current FP8 blockwise backward | `lumen/ops/quantize/linear.py:3091-3235` |
| Dense MXFP4 forward/backward | `lumen/ops/quantize/linear.py:2179-2560` |
| Dense MXFP4 cache | `lumen/quantize/__init__.py:543-653,1176-1245` |
| 32×32 weight scale contract | `lumen/ops/quantize/ops.py:619-645` |
| Standard `--linear-fp4` CLI | `lumen/patches/builders/megatron_args.py:261-327` |
| Backend warmup/sync cache behavior | `lumen/ops/dispatch.py:540-620` |
| AITER gfx950 A4W4 forward test | `aiter/ops/flydsl/test_flydsl_moe_a4w4.py:4-18,41-43` |
| AITER A4W4 stage1/stage2 | `aiter/ops/flydsl/moe_kernels.py:1099-1152,1453-1505` |
| AITER DSV4 A8W4 tuned rows | `aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv:122-137` |
| AITER gfx950 A4W4 EP limitation | `aiter/fused_moe.py:702-713` |
| AITER config key includes local E | `aiter/fused_moe.py:569-570,1770-1783` |
| AITER non-MXFP4 MoE WGrad | `aiter/ops/triton/moe/moe_wgrad.py:47-211` |
| Lumen benchmark timer/statistics | `benchmarks/bench_utils.py:91-174` |
| Lumen overlap reporting | `benchmarks/bench_utils.py:276-318` |
