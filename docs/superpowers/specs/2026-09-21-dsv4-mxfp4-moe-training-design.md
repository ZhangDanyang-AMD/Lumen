# Lumen MXFP4 A4W4 for DeepSeek-V4 Routed Experts — Training Design Spec

**Date:** 2026-09-21

**Last updated:** 2026-10-09（本轮一手源码 reconciliation R19；保留 R1–R18 历史）

**Status（分层）:** M0 frozen；P0a operator contract frozen；P0b fixed-random-init 4L training-closure/regression smoke decided; concrete run manifest pending；P1 architecture reviewed；P1 compiled ABI draft；validation protocol correctness frozen、quality/performance pending preregistration；execution authorization not granted；implementation not started

**Lifecycle:** Spec developing; Q1–Q34 review decisions incorporated; 2026-10-09 amendments R1–R6 and GPT-review amendments R7–R18 recorded

**Scope:** gfx950 上 DeepSeek-V4 routed-expert FC1/FC2 的 Lumen-native MXFP4 A4W4 experimental training path

**Primary repositories:** Lumen + AITER

**Implementation status:** Not started

本文中的 MUST / MUST NOT / SHOULD / MAY 分别表示强制要求、强制禁止、推荐要求和可选能力。未经对应验收门禁，不得把设计项描述为已支持能力。

---

## 1. 结论

固定 Lumen revision c6d2e121a81b73ef0acdfe323c11c0254d8ac7bb 的 DeepSeek-V4（下文简称 DSV4）FP8 MoE 链路是（不得外推所有版本，证据见 §3.5）：

~~~text
Megatron router / permutation / EP all-to-all / local expert sorting
  -> LumenGroupedLinear
  -> Python 逐 expert 调用 blockwise FP8 quantized_linear
  -> FC1、BF16 clamp/SwiGLU/probability、FC2
  -> Megatron unsort / combine
~~~

它不是单次 grouped FP8 training operator，WGrad 对动态 M_e 的对齐限制还会使部分 expert 回退 BF16。因此现状不能作为 grouped、全低精度 backward 或 strict zero-fallback 的证据。

本 spec 冻结的目标是：

> **Lumen-native FP4 A4W4 training for DeepSeek-V4 routed-expert shapes — experimental pretraining/continued-training.**

该名称只表示模型结构、shape 和集成位置，不表示复现或数值等价于 DeepSeek-V4 报告中的 QAT recipe。目标链路保留 Megatron 的 router、dispatch/combine、SwiGLU/clamp、routing probability gradient、checkpoint/recompute 语义，只在 expert-sorted tensor 边界替换 routed-expert FC1/FC2 的 GEMM。Router 保持原精度/BF16 且可训练；gate Parameter gradient 与 e-score bias 的非梯度状态更新分别保留，不冻结整个 router。

交付按三个阶段推进：

1. **M0 / prerequisites:** 从 Lumen、AITER dirty workspace 中抽取最小必要前置提交，逐项测试并建立唯一 clean lineage。
2. **P0 / sequential:** 使用与 P1 完全相同的数值 recipe、layout、RNG、cache 和输出 dtype，逐 expert 建立 correctness reference。P0 拆分为：**P0a operator reference**（sequential A4W4 算子契约与数值/生命周期门禁，本 spec 已冻结）与 **P0b training closure/regression smoke**（4L 分布式训练 fixture 与数据/配置冻结，设计选择已关闭，具体运行 manifest 待冻结）。
3. **P1 / grouped:** AITER 提供 grouped Fprop/DGrad/WGrad，Lumen 提供 autograd、事务式绑定、cache、optimizer/checkpoint 生命周期、dispatch、ledger 和 Megatron 集成。

最终 promotion 不是“能训练”或“loss 有限”。它要求 strict zero-fallback、全拓扑/恢复门禁、预注册质量统计、真实 routing bucket certification 和端到端性能门槛同时通过。即使全部通过，V1 仍是 opt-in experimental、仅对已测试配置有效，不称通用 production/convergence parity。

---

## 2. 定位、允许声明与非目标

### 2.1 产品定位

V1 的正式名称固定为：

~~~text
Lumen MXFP4 A4W4 for DeepSeek-V4 routed experts
~~~

允许的最高级别声明是：

~~~text
experimental, opt-in, qualified for the exact tested
DSV4 routed-expert recipe/topology/horizon on gfx950
~~~

在完成第 17 节的统计门禁前，只能称为 stability 或 short-horizon quality-regression 结果。V1 不得声称：

- DSV4 QAT reproduction；
- 与 DSV4 报告数值等价；
- convergence parity；
- production quality；
- downstream quality parity；
- 任意 topology、任意 shape 或任意 GPU 支持。

### 2.2 V1 scope

V1 MUST 覆盖：

- GPU architecture：gfx950；
- expert tensor parallel：ETP=1；
- EP8 / local-E32；
- EP4 / local-E64；
- FC1：[M,4096] x [4096,4096]；
- FC2：[M,2048] x [2048,4096]；
- Fprop、DGrad、WGrad；
- 动态 expert token 数、empty expert、负载倾斜和 extreme token count；
- gradient accumulation；
- full activation recompute；
- warm_start、exact_continue 和 reshard_continue 的明确语义。

V1 MUST NOT 扩展到：

- gfx942/gfx1250；
- ETP>1；
- router、routing logits/probabilities、dispatch/combine 通信量化；
- shared expert、attention、embedding、lm-head；
- 训练化完整 AITER inference fused_moe；
- deferred/delay WGrad；
- 动态 routing 的端到端 CUDA graph replay；
- FP4 checkpoint 主表示、FP4 optimizer state 或 rollout/export；
- 把 A8W4 或 BF16 当作 strict A4W4 的静默 fallback。

OQ-G-01 已关闭：V1 明确排除 FSDP，仅覆盖 Megatron-Core expert training path；reshard_continue 仅指 Megatron/DCP topology reshard，不含 FSDP flat-parameter lifecycle。公共 module 保持 Parameter/state_dict 语义；TE FSDP 生命周期仅为未来接口参考，不进入 V1 exit criteria。

### 2.3 DSV4 报告与本设计的差异

DSV4 报告的 MoE 数值链路是：

~~~text
FP32 optimizer master
  -> MXFP4 E2M1 weight, 1x32 scale
  -> exact FP4-to-E4M3 representation
  -> FP8 activation x FP8 effective weight
  -> STE to FP32 master
  -> same native packed FP4 weight for rollout
~~~

本设计则是：

~~~text
BF16 model Parameter as quantization source
  -> weight E2M1 + E8M0 tile32x32/S1024
  -> Fprop A4 x W4
  -> DGrad dY4 x W4-transpose-view
  -> WGrad dY4^T x X4
  -> FP32 main_grad accumulation
~~~

两者的 activation precision、weight scale granularity、训练 kernel、gradient quantization 和 rollout/export contract 均不同。DSV4 的 S32/A8W4 路径若实现，MUST 使用独立 recipe/spec，不能作为本设计的别名或运行时选项。

官方报告（[arXiv:2606.19348v1](https://arxiv.org/abs/2606.19348v1)）明确其 FP4 QAT 在 **post-training stage** 引入（其 [§5.2.1](https://arxiv.org/html/2606.19348v1#S5.SS2.SSS1) 与 [§1 Introduction](https://arxiv.org/html/2606.19348v1#S1) 有直接原文），作用于 teacher/reference 模型；其整体预训练按标准 from-scratch 语义描述，但未明示随机初始化（未证实，不得写成已证实）。本设计是独立的 A4W4 experimental pretraining/continued-training：既不声称复现官方 QAT，也不援引官方报告作为 MXFP4 从头预训练可行性的证据。

---

## 3. 审计基线与 clean-lineage 要求

### 3.1 已审计 snapshot

原始源码审计基于以下 snapshot：

| Repository | Branch / source | Commit |
|---|---|---|
| Lumen | dev/mxfp4 audit snapshot | 9d85c8adb5159cc5765680bbc4c2230bb00e74e4 |
| AITER runtime checkout | bench/ecfff3f-lumen | e35bb17f4f815903bf73598facedbb321e15af28 |
| Megatron-LM | core_r0.15.0_rocm | 1b754411c7777799fb17e260baa7145c0c6d71a2 |
| Lumen AITER gitlink | dependency pointer | ecfff3fa80f906c5c421a35a7f5e52842f000559 |

runtime AITER 与 Lumen gitlink 不一致，且 Lumen/AITER workspace 均含未提交改动。该 snapshot 只可用于需求审计，不可用于正式 correctness、quality 或 performance certification。

### 3.2 M0：唯一 clean lineage

任何实现前 MUST：

1. 分别列出 Lumen、AITER 的 tracked 与 untracked dirty diff。
2. 只抽取 MXFP4 MoE 所需的最小前置改动；不得把整个 dirty workspace 打成一个 baseline commit。
3. 每个前置提交独立审查、独立测试，记录它解决的具体 prerequisite。
4. 固定 Lumen、AITER、Megatron、ROCm、compiler、PyTorch 和容器版本。
5. 使 Lumen gitlink、实际 import 的 AITER source 和实际加载 extension build-id 指向同一 lineage。
6. 2026-10-09 用户批准的 upstream-first 修订（§23 R2）：AITER clean baseline 直接建立在当前 upstream main 的明确 commit 上（拟建独立 worktree `/scratch/users/xdai/aiter-mxfp4-moe`，本记录时不创建），旧恢复 dirty 仅作逐项语义来源并保留；取代旧"先在旧 M1 完成 P0/P1、ABI 冻结后再 forward-port"的顺序。ABI 冻结后若 upstream 前移，forward-port 到新的明确 pinned commit 仍作为独立 gate，并重复 conformance、kernel correctness 和最小训练门禁。

正式 manifest 至少包含：

~~~text
Lumen/AITER/Megatron commit
Lumen AITER gitlink
source tree/content digest
dirty=false
aiter.__file__ / lumen.__file__
loaded .so or wheel digest and build-id
ABI version / training/operand-quant/operand-transform recipe IDs / layout IDs
ROCm / driver / compiler / PyTorch / RCCL
GPU arch / CU count / topology / rank mapping
build flags / tuning database digest
~~~

仅检查工作区路径或 git commit 不足以证明实际加载的二进制与源码一致。

### 3.3 当前链路的关键事实

历史 snapshot 审计结论如下；其中 FP8 sequential/backward 以本轮 §3.5 固定 revision 重新核对，AITER 能力以 §3.6 矩阵限定，不将历史 snapshot 泛称所有当前版本：

- Megatron 已正确拥有 router、token permutation/local sorting、EP all-to-all、SwiGLU/clamp、combine、router gradient 和 recompute。
- 固定 Lumen c6d2e121 的 LumenGroupedLinear 按 expert Python loop 调 quantized_linear，不是 grouped GPU 调度。
- 默认 DSV4 blockwise FP8 的 Fprop/DGrad 可低精度执行，但 WGrad 对 M_e % 128 的限制会触发 BF16。
- DSV4 的 post-build FP8 enable 没有使 Megatron config.fp8 为真，因此不能假定 Megatron FP8 padding 已启用。
- dense MXFP4 已有可复用的 A4W4、dual-layout、H16、SR 和 weight-cache 能力，但不能直接证明 grouped MoE correctness/performance。
- AITER 现有 DSV4 tuned rows 是 A8W4 inference rows，且 local-E/训练 backward contract 不匹配。
- 本轮固定 AITER 663f69b2 的已核对公共路径不提供本 spec 完整的 grouped MXFP4 F/D/W 训练合同；已有能力和具体缺口见 §3.6，不能概括成“没有 grouped GEMM”。
- 当前 Python random 生成 SR seed/offset 的路径不满足 exact replay，MUST 移除出本链路。
- distributed checkpoint 可原位恢复 Parameter bytes，而 Parameter identity 与 _version 不变；因此 _version 不能作为 cache correctness authority。

### 3.4 Current FP8 MoE evidence map

| Observation | Source area | Consequence |
|---|---|---|
| DSV4 provider keeps Megatron grouped-expert shell and substitutes Lumen FC1/FC2 | lumen/models/dsv4/megatron/spec_provider.py | integration boundary can remain at sorted tensors |
| Grouped module loops experts and calls quantized_linear separately | lumen/modules/grouped_linear.py | current path is sequential, not grouped execution |
| Current grouped GEMM op has no MXFP4 training branch | lumen/ops/gemm/grouped_gemm.py | P1 needs a new AITER public operator and Lumen adapter |
| DSV4 enables FP8 after model construction | lumen/models/dsv4/megatron/fp8.py and pretrain.py | Megatron config.fp8 padding cannot be assumed |
| Blockwise backward re-quantizes a transpose and WGrad has token-axis alignment constraints | lumen/ops/quantize/linear.py | dynamic M_e can trigger BF16 WGrad |
| Dense MXFP4 has reusable F/D/W and cache pieces | lumen/ops/quantize/linear.py, lumen/ops/quantize/ops.py, lumen/quantize/__init__.py | reuse only after numeric/lifecycle conformance |
| Existing AITER DSV4 table is FP8-activation/FP4-weight inference | aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv | it is not an A4W4 training tuning source |
| Existing AITER MoE WGrad is not the required MXFP4 operand contract | aiter/ops/triton/moe/moe_wgrad.py | new or extended public training API is required |

---

### 3.5 Fixed Lumen evidence — sequential is not grouped GPU execution

本轮读取的 Lumen 固定 SHA 为 `c6d2e121a81b73ef0acdfe323c11c0254d8ac7bb`，与承载本设计文档的 `7c98d87` 分支基点分开记录：

- [grouped_linear.py L126–167](https://github.com/ZhangDanyang-AMD/Lumen/blob/c6d2e121a81b73ef0acdfe323c11c0254d8ac7bb/lumen/modules/grouped_linear.py#L126-L167)：`m_splits[i]` → expert slice → dense `quantized_linear` → `torch.cat`；empty expert 跳过，all-empty 返回空 tensor。`m_splits_gpu` 仅在签名出现，未用于 grouped GPU 调度。
- [dense backward L2351–2533](https://github.com/ZhangDanyang-AMD/Lumen/blob/c6d2e121a81b73ef0acdfe323c11c0254d8ac7bb/lumen/ops/quantize/linear.py#L2351-L2533)：dense autograd 已有 DGrad/WGrad。blockwise DGrad 对 N/K 对齐，M 不受此限制；WGrad `_aligned_wgrad=(M % block_size == 0)`，非对齐 M_e 使 `g_col=None` 后执行 `_bf16_wgrad()`；kernel rejection 也可触发 BF16。不能把“有 backward”误写成“全 grouped/全低精度 backward”。当前 WGrad FP8 分支实际调用 `requant_fp8_row_to_col`，不是注释描述的必然 BF16 中间张量路径。
- [fp8.py L10–80](https://github.com/ZhangDanyang-AMD/Lumen/blob/c6d2e121a81b73ef0acdfe323c11c0254d8ac7bb/lumen/models/dsv4/megatron/fp8.py#L10-L80)：DSV4 默认 blockwise/block128，通过 model 构建后的 enable hook 配置；此 hook 不设置 Megatron `config.fp8`，不可据此假设 framework FP8 padding。

这些事实只针对该 revision。P0a 使用逐 expert **dense MXFP4** F/D/W parity，修补/验证其 frozen recipe、empty/tiny/nonaligned/skew 和生命周期，不把旧 FP8 fallback 当作通过；P0b 是显式 sequential correctness backend 的固定随机 4L smoke；P1 才针对已有能力逐项复用/扩展，交付真正满足冻结合同的 AITER grouped MXFP4 F/D/W 与 Lumen autograd/cache/dispatch，而非从零复制所有 grouped GEMM。

### 3.6 AITER capability-gap matrix — current source outranks historical issues

本轮通过 `git ls-remote` 固定 upstream main 为 `663f69b223cdc231c7b2c20d1104aa6626b899b9`；这是证据 pin，不是已安装/测试或获准实现的 baseline。[Issue 3038](https://github.com/ROCm/aiter/issues/3038) 正文提出 training TN/NN/NT、routing-free splits、caller out/accumulate、scale/empty 合同；[roadmap 3443](https://github.com/ROCm/aiter/issues/3443) 把它列为 training planned direction。两者是需求/历史状态，不能独自证明本次源码缺失。

| 已读取固定源码 | 已有能力 | 对本 V1 的具体缺口/处理 |
|---|---|---|
| [MoE a8w8](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/moe/moe_op_gemm_a8w8.py) | FP8/MX scale forward、RoutingData/gather/scatter、SwiGLU/reduction；自行分配输出 | 不提供同一 training F/D/W、caller main_grad accumulate 合同；scale transpose/recipe 必须独立验证 |
| [gmm / ptgmm](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/gemm/grouped/gmm.py) + [dtype validation](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/utils/gmm_common.py) | BF16/FP16 grouped F/D、reduction-axis WGrad；device group_sizes；existing_out；ptgmm accumulate | 不是 FP8/MXFP8/MXFP4 packed-scale 合同；不能说 AITER 完全没有 grouped training 或 caller out |
| [SonicMoE wrapper](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/moe/sonicmoe.py#L66-L316) + [kernel](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/_triton_kernels/moe/sonicmoe/grouped_gemm.py) | **已有** `grouped_gemm(A,B,cu_seqlens,out,...)`、transpose flags、blockwise FP8 F/D 与 `_grouped_gemm_dw`；可不传 gather/scatter，caller out 可用；kernel 直接 store | 部分超越 3038 的历史“仅 forward”说法。scale 固定 128（activation1x128/weight128x128；WGrad token-axis scales），没有 MXFP4/E8M0/H16-SR 合同；无 accumulate 参数，[WGrad kernel L243–319](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/_triton_kernels/moe/sonicmoe/grouped_gemm.py#L243-L319) store 覆盖而非 main_grad +=；[wrapper L243–276](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/moe/sonicmoe.py#L243-L276) scale validation 使用 `.sum().item()`，不符合 P1 新增 D2H 禁令。应优先复用/扩展而非重复 kernel；不因接口存在宣称完整稳定通过 |
| [moe_wgrad](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/moe/moe_wgrad.py) | 已有 sorted-token WGrad、empty sorted input 返回 zero dW | 依赖 sorted ids/top_k，内部 zeros allocation、没有 packed scales/caller accumulation；不等于所需 MXFP4 WGrad |
| [dense FP4 GEMM](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py) | packed E2M1 + E8M0 dense GEMM，caller y、split-K、prepared layouts | dense 非 grouped；caller y 不等于 FP32 main_grad accumulate；其每行1x32 scale 接口不得无审计当作 Lumen weight32x32 recipe |
| [FlyDSL MoE](https://github.com/ROCm/aiter/blob/663f69b223cdc231c7b2c20d1104aa6626b899b9/aiter/ops/flydsl/moe_kernels.py) | FP8/MXFP8/FP4 前向 stage1/2、sort/top-k/EP reduction | stage1 FC1–SwiGLU 与 stage2 FC2–reduction 是两个 **forward** 阶段，不是 forward/backward；不能替代训练 D/W |

另一个可核实的历史冲突：3038 写 a8w8 仅 scalar static scales；本次固定 wrapper 已有 `x_token_scale` 和 `w_expt_scale`，不得继续宣称该项完全缺失。SonicMoE 以上结论来自公开 Python wrapper 与其内部 kernel 分支，并非全组合 GPU 测试证明：未审计/运行的 dtype-layout 组合与 empty/ragged 稳定性不得标为 supported。当前缺口是**满足冻结 MXFP4 recipe 的完整稳定训练 bundle**，不是所有低精度 grouped 算子不存在。A0 必须逐 pass 对 TN Fprop / NN DGrad / NT WGrad、routing-free splits、caller output/main_grad overwrite/accumulate、empty/all-empty/ragged 与 determinism 给出 capability 和测试，已有局部支持不得冒充全 bundle；本轮未运行任何 GPU 测试。

### 3.6.1 Dense MXFP4 candidate reuse — pass-by-pass audit prerequisite

用户已提交的 dense 算子是候选复用来源，不代表完整 MoE 契约已支持。M0/P0a MUST 记录真实 Lumen/AITER commit、public API/symbol、实际调用链与测试证据；§3.1/§3.5 历史 pins 不能自动替代实际候选 pin。

| Pass / item | 复用前必须证明的合同 | 未通过时处理 |
|---|---|---|
| Fprop | X4 @ W4^T 方向；E2M1 packed payload/E8M0 scales；activation1x32 与 weight32x32/S1024 的 axes/layout/selector；FP32 accumulation 到 BF16 | 记录具体 dense capability gap，优先补 dense 公共合同 |
| DGrad | dY4 @ W4-transpose-view 方向；同 forward 权重 snapshot/scale bytes 的确定性 repack；dY SR token；禁止更新后重量化或启用 DGrad Hadamard | 不宣称仅有 dense backward 就已满足冻结 recipe |
| WGrad | dY^T @ X；两 operand 先逻辑零 pad 再 normalized H16；RTN/SR、dual layouts 与 logical RNG；OVERWRITE/ACCUMULATE_FP32，直接 FP32 main_grad 且 exactly-once | caller output 不等于 FP32 accumulate，禁止 BF16 中间落地冒充通过 |
| 所有 pass / lifecycle | FC1/FC2 empty/all-empty/tiny/nonaligned/ragged/skew、cache-disabled parity、recompute/optimizer/cache epochs、kernel_family allowlist 与 noCK（含内部 dispatch） | 未审计/未测试组合保留 unsupported/待验证；strict typed fail，不能静默 BF16 |

P0 缺口优先修补 dense 契约与 Lumen adapter，不为建立 sequential correctness backend 新造 grouped kernel；P1 后续按 §3.6 grouped capability-gap 逐项扩展。

### 3.7 DeepGEMM: API/scheduler and test-design reference only

本轮 main 固定为 `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`，读取 [gemm.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/gemm.hpp)、[test_fp8_fp4.py](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/tests/test_fp8_fp4.py) 和 [generators.py](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/tests/generators.py)。测试已重构，不沿用用户所列旧行号，以下以实际 symbol 定位。

| API | 实际数学/输入与输出边界 | 本设计参考点 |
|---|---|---|
| `m_grouped_fp8_fp4_gemm_nt_contiguous(a,b,d,grouped_layout,...)` | `[M,K] @ [G,N,K].mT`，caller BF16 d、int32 PSUM 或 per-row group layout，M=0 early return | Fprop M-grouped ragged descriptor/scheduling；注意其 NT 名称与 TE 调用 weights-first TN 不同，不机械按字母对齐 |
| `m_grouped_fp8_fp4_gemm_nn_contiguous(...)` | 包装器转置 B payload/scale 再调 NT | DGrad API/layout 参考；测试生成器对 FP4×FP4 的 MN-major B 组合跳过，不能凭 NN symbol 存在宣称 native A4W4 DGrad 全支持 |
| `k_grouped_fp4_gemm_nt_contiguous(a,b,d,ks_cpu,grouped_layout,c,... )` | 逻辑 A `[m,sum_k]`、B `[n,sum_k]`（packed 最后一维减半、K-major），D `[G,m,n]`，可选 C；recipe `(1,1,32)`，k_alignment 为256倍数，SM100 路径 | WGrad `dY^T[N,M_e] @ X[M_e,K]`，映射 m=N、n=K、每组 reduction k=M_e；必须参考 **K-grouped** reduction 调度，不能机械套 Fprop 的 M-grouped |

K-grouped C/D 可别名做累加；总 K=0 的 early-return 对 C==D 保持、不别名复制 C、无 C 清零 D。测试包含 partial-empty/首组empty/single-hot、PSUM/host-K variants、BF16/FP32 overwrite、C/D accumulate、重复 deterministic 检查；未见显式 all-empty generator case，API early return 不等于已执行 all-empty GPU 测试。Lumen 必须补其 own empty OVERWRITE 清零 / ACCUMULATE 不变测试。

只借 API、ragged、reduction、C/D accumulation 与 empty 测试设计：不是实现依赖，不可在 AMD/gfx950 直接调用，不是 Lumen 数值 oracle，也不证明 DSV4 历史内部使用了 current main。其 FP4 1x32 scale/layout 与冻结 activation1x32、**weight32x32**、H16-RHT/SR 并不默认等价，禁止移植其量化 recipe 替代 §5。

### 3.8 Forward fusion, PDL and TransformerEngine boundaries

[DeepGEMM PR304](https://github.com/deepseek-ai/DeepGEMM/pull/304) 正文明确 MegaMoE 融合 dispatch/linear1/SwiGLU/linear2/combine，**Only FP8 × FP4**，且声明该 release 与 internal model release 无关；对应 [固定 release README](https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/README.md) 也给出 `fp8_fp4_mega_moe` forward。它属于 A8W4 fused forward/rollout/inference 方向，不提供 DGrad/WGrad 证据；与 §3.7 独立 grouped weight-backward API 区分。

[NVIDIA CUDA PDL 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#programmatic-dependent-kernel-launch)（本轮读取 v13.4.2）将 PDL 定义为 **Programmatic Dependent Launch**：primary trigger、secondary grid dependency synchronize 和 launch attribute 允许同 stream dependent kernels 部分重叠。它不是 expert wave 调度概念，不是 gfx950 V1 capability 或依赖。

TE main 本轮固定 `39a8b879dfbfea0d23bc3862dc83ec1d6f970b49`，读取 [grouped_linear.py](https://github.com/NVIDIA/TransformerEngine/blob/39a8b879dfbfea0d23bc3862dc83ec1d6f970b49/transformer_engine/pytorch/module/grouped_linear.py)。仅用作 Lumen §9/§11 封装与生命周期参考：

- saved operands / offsets 经 prepare_for_saving 保存，row/column usage 按 F/D/W consumer 准备；weight workspace/cache 受 first-microbatch 与 update 控制。
- forward TN、DGrad NN、WGrad NT 调用与 caller output、main_grad accumulate/overwrite 分开；weakrefs 指向原 Parameter 而非临时 gathered copy。
- FSDP lazy main_grad accessor、distributed weight backward re-materialization、finalize grads，提示未来 sharding 生命周期；TP 通信由 MoE dispatch/combine 处理而非模块内部复制。**不将 FSDP 加回 V1 scope。**
- 防重复梯度包含 `grad_added_to_main_grad` bookkeeping 与 None/dummy-grad 约定；Lumen 应审查自己的 Megatron reducer 合同，不能机械照抄 TE dummy grads 或产生第二份真实梯度。
- TE NVFP4 并非 OCP MXFP4；不复制其 block16、E4M3/global scale 或 RHT recipe。仅凭 wrapper 的 NVFP4 开关不能改写 Lumen §5 的冻结格式、32x32 weights 和 H16-SR。

---

## 4. 固定 shape、边界与数据流

### 4.1 Shape

对本地 expert e：

~~~text
E_l       local expert count, 32 or 64
M_e       routed rows for expert e
M         sum(M_e)
P_e       ceil(M_e / 32) * 32
K         input width
N         output width
~~~

| Role | Input | Weight | Output | K | N |
|---|---|---|---|---:|---:|
| FC1 gate+up | [M,4096] | [E_l,4096,4096] logically | [M,4096] | 4096 | 4096 |
| FC2 down | [M,2048] | [E_l,4096,2048] logically | [M,4096] | 2048 | 4096 |

FC1 前半/后半继续按 Megatron 现有 gate/up 顺序解释。

### 4.2 Integration boundary

MXFP4 层只消费：

~~~text
x_sorted
group_sizes / expert_offsets
expert weights
~~~

端到端路径固定为：

~~~text
BF16 hidden states
  -> Megatron router/top-k
  -> local permutation
  -> EP all-to-all
  -> local expert sorting
  -> selected sequential/grouped MXFP4 FC1
  -> existing BF16 clamp + SwiGLU
  -> existing routing-probability multiply
  -> selected sequential/grouped MXFP4 FC2
  -> local unsort
  -> TP/EP combine
~~~

MXFP4 implementation MUST NOT 接收或重新解释 global router logits、top-k ids、top-k weights 或 expert mask。Probability gradient、dispatch/combine 和 checkpoint keys 继续由 Megatron/Lumen integration layer 拥有。

P0b 数据合同：固定随机初始化 4L，BF16/MXFP4 必须共用相同 model/optimizer/seed/data 顺序、attention masks、positions 和 loss target token 集合。Packing MUST 强制跨文档 attention 隔离；用显式文档边界、有效长度与 target mask 区分 padding/真实 EOS，EOS=pad 时 MUST NOT 仅凭 token ID 屏蔽真实 EOS。验收保存两臂 valid-target 与 mask/position digest，并加入两篇文档 sentinel、真实 EOS 与尾部 padding 同 ID 的负向测试。具体 seed/超参数值未给出，不据此发明已批准值。

### 4.3 Fprop/DGrad/WGrad

~~~text
Fprop: X4 [M_e,K] x W4 [N,K]^T -> BF16 Y [M_e,N]
DGrad: dY4 [M_e,N] x W4-transpose-view -> BF16 dX [M_e,K]
WGrad: Q(H16·pad(dY_e))^T [N,P_e] x Q(H16·pad(X_e)) [P_e,K] -> FP32 accumulation dW_e [N,K]
~~~

WGrad 的两个 operand 使用同一 H16 transform（同一 operand_transform_id、同一 normalized 约定、同一 diagonal sign vector）；语义细则见第 5.4 节。

P0 sequential 与 P1 grouped MUST 使用完全相同的 quantizer、scale bytes、physical layouts、RTN/SR、padding、accumulator 和 output dtype。P0 是 executable numerical reference，不是另一套 recipe。

---

## 5. 数值 recipe 与格式版本

### 5.1 权威 ID 分层

不得使用 canonical MXFP4 作为含混格式名。V1 将训练 recipe、operand 量化数值、operand 数值变换与物理布局分成独立演进的层：

~~~text
training_recipe_id
operand_quant_recipe_id
operand_transform_id
physical_layout_id
prepared_layout_id
~~~

training_recipe_id 标识整体训练 recipe 标签：

~~~text
lumen_a4w4_direct_training_v1
lumen_w4_s1024_a8_nonweight_operands_v1
~~~

前者是本 spec 主线 A4W4 直接训练；后者是第 5.6 节 matched-scale A8W4 质量控制（既有 ID 保留）。

operand_quant_recipe_id 标识单 operand 的量化数值 recipe。V1 固定两个，命名按 scale block 粒度而非物理 tile，避免与 physical_layout_id 混淆：

~~~text
lumen_fp4_e2m1_e8m0_scale_block_32x32_s1024_v1
lumen_fp4_e2m1_e8m0_scale_block_1x32_v1
~~~

角色（2026-10-09 修订经 dense MXFP4 源码核对后澄清）：

- scale_block_32x32_s1024：weight operand 的 2D tile scale recipe（每 32x32 tile 共享一个 E8M0 scale，1024 value/tile），对应 dense `convert_to_mxfp4_2d` 语义；
- scale_block_1x32：非 weight operand（X、dY row 与 WGrad transposed operand）的 per-row 1x32 scale recipe，对应 dense `convert_to_mxfp4` / `hadamard_quant_mxfp4` 的 scale 粒度。旧文把两个 ID 并列于"主 weight numeric recipe"不准确：scale_block_1x32 不是第二个 weight recipe，也不是 DSV4 S32 weight recipe 的别名；DSV4 S32 路径仍按第 2.3 节要求使用独立 recipe/spec。

每个 operand_quant_recipe_id MUST 完整规定：

- E2M1 payload 与 E8M0 scale；
- scale block 粒度（32x32/S1024 或 1x32）和 axes；
- fp4_e2m1_e8m0_selector_v1 的 amax、exponent floor/ceil、tie、encode/clamp；
- RTN 或 SR（按 operand role，第 5.3/5.4 节）；
- zero、signed zero、NaN、Inf、subnormal、overflow 和 saturation；
- byte-exact golden vectors。

operand_transform_id 标识量化前的数值变换，不是 physical layout：

~~~text
hadamard16_all_plus_v1
~~~

H16 从旧 physical layout 列表改列到本层（2026-10-09 修订）：它改变 operand 的数值分布，不改变 packed byte 的物理排布；语义细则见第 5.4 节。

physical_layout_id MUST 规定 nibble order、byte encoding/endianness、tile traversal、strides、padding、scale placement、offset/pointer width 和最低 alignment；不再承担 H16 数值变换语义。

V1 公共 physical layout IDs 固定为：

~~~text
mxfp4_weight_tile32x32_v1
mxfp4_segmented_row_and_t_v1
~~~

prepared_layout_id 只标识 kernel-private swizzle/preshuffle。它不是公共 interchange format，也不允许改变数值值域或重新量化 BF16 source。

ABI version、training/operand-quant/operand-transform recipe version、physical layout version 和 prepared layout version MUST 独立演进。破坏性 ABI 改动发布 _v2。

### 5.2 Weight source 与双布局

V1 固定：

~~~text
weight_quant_source=model_parameter_bf16_v1
~~~

生命周期是：

~~~text
FP32 optimizer master update
  -> sync to BF16 model Parameter
  -> readiness event completes
  -> atomically advance committed_weight_epoch
  -> Q4(BF16 Parameter)
~~~

必须公开承认：

~~~text
Q4(cast_bf16(W_fp32)) != Q4(W_fp32)
~~~

Forward weight 与 DGrad transpose view MUST 来自同一 BF16 snapshot 和同一量化结果。DGrad 可以使用确定性的 transpose/repack/view，但 MUST NOT 从更新后的 BF16 weight 重新量化，也 MUST NOT 用不同的 scale selector。

Model/checkpoint Parameter 为 BF16，optimizer master 为 FP32，production WGrad 直接累加 FP32 main_grad。Optimizer moments 的 dtype 沿用对应 BF16 baseline，并作为 manifest 字段；MXFP4 不得静默改变 optimizer state dtype。

### 5.3 Forward activation 与 WGrad activation

对同一 BF16 X_e snapshot 独立生成：

~~~text
X_e_bf16
  +- RTN -> row-packed FP4 for Fprop
  +- H16 / transpose / logical-zero pad -> RTN -> FP4 for WGrad
~~~

禁止从已量化 row operand decode 后再量化 transposed operand。Aligned/ragged shape MUST 具有同一 reference 数学定义；pre-transform 逻辑零 padding 不参与 amax，padded lane 固定 +0；H16 之后的 post-transform lanes 是普通变换值，与真实行同权参与 amax、量化与 GEMM。

### 5.4 Gradient quantization

对 dY：

- DGrad row operand：1x32 E8M0，E2M1，SR；
- WGrad transposed operand：per-expert H16/transpose/pad，E2M1，SR；
- 两者使用不重叠 RNG subrange；
- LUMEN_MXFP4_DGRAD_HADAMARD=1 在本模式 MUST 被拒绝；
- recompute/physical retry MUST 复用逻辑 invocation 的同一 RNG token。

H16（hadamard16_all_plus_v1）语义固定为：

- X 与 dY 的 WGrad transposed operand 使用同一 H16：同一 16x16 Hadamard 矩阵、同一 normalized 约定、同一 diagonal sign vector；
- all_plus 指 diagonal sign vector 全正（+1），即 transform 是纯 Hadamard 旋转、无随机符号翻转；
- normalized 约定必须显式：V1 采用 normalized H16（H/√16，entries ±1/4，与 dense 实现一致），此时 H^T·H = I，WGrad 无需额外幅值补偿；任何 kernel 内部若改用 unnormalized ±1 整数矩阵，MUST 在数值 contract 中显式记录 1/16 compensation 的施加位置（unnormalized 时 H^T·H = 16·I）；
- padding 顺序固定：先在 BF16 域补逻辑零（pre-transform padding lanes 固定 +0、不参与 amax），再施加 H16；post-transform lanes 是普通变换值，与真实行同权参与 amax、量化与 GEMM；该顺序 MUST NOT 颠倒，禁止先变换后补零。

### 5.5 WGrad output mode

低层 WGrad contract 使用枚举，不使用 accumulate: bool：

~~~text
OVERWRITE
ACCUMULATE_FP32
~~~

语义：

- OVERWRITE：所有 output buffers 必须统一为 BF16 或统一为 FP32；M_e=0 时清零对应 expert output。
- ACCUMULATE_FP32：所有 output buffers 必须为 FP32，直接执行 main_grad_e += dW_e；M_e=0 时保持 buffer 不变。

Production path MUST 直接写 FP32 main_grad。禁止先落 BF16 再 cast/add。每个 output 的 shape、stride、device、dtype、capacity、alignment 和 alias 均在 bind/attach 前验证。

### 5.6 A8W4 controlled ablation

正式控制 recipe 名称：

~~~text
lumen_w4_s1024_a8_nonweight_operands_v1
~~~

它是 matched-scale format lift：

~~~text
Weight:      E2M1 + E8M0 tile32x32/S1024, unchanged
Non-weight:  E4M3FN + the same E8M0 bytes selected by
             fp4_e2m1_e8m0_selector_v1 over each BF16 group
~~~

逐 pass：

~~~text
Fprop: Q8_RNE(X)    x Q4(W)   -> FP32 acc -> BF16
DGrad: Q8_SR(dY)    x Q4(W)^T -> FP32 acc -> BF16
WGrad: Q8_SR(dY)^T  x Q8_RNE(X) -> FP32 main_grad
~~~

因此 A8W4 是整体训练 recipe 标签；WGrad 本身没有 W4 operand。

这里 RNE 是 V1 对 RTN 的精确定义：round-to-nearest, ties-to-even。E4M3FN 的 finite max、encoding、special-value 和 saturation 由版本化 codec 与 golden vectors 固定；FNUZ、E5M2、按 FP8 max 重新选择 scale 或 per-row FP32 scale 都不属于该 control。

Fixed-snapshot paired operator/one-step test MUST 对同一 BF16 group 只执行一次 A4 selector，并把同一份 E8M0 scale bytes 同时交给 E2M1 和 byte-exact-defined E4M3FN quantizer。Full-training arms 只共享初始 checkpoint、optimizer/scheduler/data order 和 recipe；首次 update 后允许各自权重与 routing 轨迹自然分叉。

Confirmatory quality ablation 中 A4/A8 MUST 使用相同 execution structure 和 strict zero-fallback。若 A8 只有 sequential reference 而 A4 使用 grouped，它只能是 diagnostic，不得进入 confirmatory non-inferiority。

A8W4 不进入运行时 fallback，不进入 A4W4 性能主表；它是必做质量消融。现有 generic FP8 路径只能复用底层组件，不能未经 contract 对齐直接作为该 control。

---

## 6. Routing metadata 与生命周期

### 6.1 Ownership

- Lumen/Megatron 拥有 routing counts 的语义、permutation generation、对象生命周期和 autograd retention。
- AITER 定义 metadata 的物理 schema 和 builder/validator。
- buffers 由 caller 拥有；AITER MUST NOT 隐藏全局 routing object。

每个 routing epoch 构造一次 immutable RoutingMetadataV1，供 FC1、FC2、DGrad 和 WGrad 复用。至少包含：

~~~text
counts
offsets
total_rows
nonempty
max_Me
capacity_rows
segment_alignment
padded_counts
padded_offsets
total_padded
routing_epoch
dispatch_generation
layout_id
~~~

不同 operand/pass 若需要不同 M-padding，MUST 按 layout_id 保存不同 padded offsets。不得假定一组 offsets 服务所有 pass。

### 6.2 Empty/ragged semantics

- M_e=0 产生合法 empty descriptor，不启动 0-M kernel。
- all-empty microbatch 对 Fprop、DGrad 和 WGrad ACCUMULATE_FP32 是 no-op；ledger terminal state 为 EMPTY，selected GEMM backend 为 none。
- WGrad OVERWRITE 是例外：每个 M_e=0 的对应 output 必须保证清零。all-empty 时可以执行已绑定的 zero-fill child operation，但不得启动 0-M GEMM；ledger 仍记录 EMPTY，并单独记录 output initialization，而不能把它伪装成 grouped GEMM success。
- 每个 expert 独立 padding，不能跨 expert 共享 quant block、scale、H16 segment 或 RNG logical index。
- metadata correctness identity 是被 BoundExecutionV1 和 autograd context 强引用的 immutable object；routing_fingerprint 只用于诊断，不能作为 correctness identity。
- 对象生命周期持续到最后一个异步 consumer 的 completion event；Python backward 返回不是生命周期终点。

### 6.3 Stable raw descriptor

稳定 descriptor 与可演进 bucketizer 分离：

~~~text
route_descriptor_version = seghist-v1
route_bucketizer_version = artifact-defined version
~~~

输入 counts 的口径固定为：top-k expansion、capacity/drop、EP dispatch、local expert mask 和 local sorting 之后的语义有效行；不含 kernel padding 或预分配 capacity。

seghist-v1 保存：

~~~text
E
total_rows
nonempty
max_rows
padded_rows32
tiles32
hist16[8]
~~~

其中：

~~~text
padded_rows32 = sum(ceil(M_i / 32) * 32)
tiles32       = sum(ceil(M_i / 32))
~~~

对 s_i = ceil(M_i / 16)，hist16 固定八个整数区间：

~~~text
s=0, s=1, s=2, s=3..4,
s=5..8, s=9..16, s=17..32, s>=33
~~~

workload size 主轴 MUST 使用实际 padded_work_bucket。不同 pass/op variant 若 work quantum 不同，MUST 有各自 bucketizer。nonempty < 10 时不得依赖不稳定 q90；使用 histogram cumulative concentration 或 dedicated small-active class。

最多 12 个 bucket 是配置数量目标，不是预先冻结的分类语义。Bucketizer MUST 由真实 artifacts 的 collision/regret audit 生成。

padded work 的计算不得依赖待选择的 kernel tile。Bucketizer artifact 对每个 op_variant 固定 candidate-independent work_quantum_rows：

~~~text
Fprop/DGrad v1: work_quantum_rows = 16
WGrad v1:       work_quantum_rows = 32
padded_work = sum(ceil(M_i / work_quantum_rows) * work_quantum_rows)
~~~

WGrad 的 32 与语义 pad32 一致；Fprop/DGrad 的 16 与 seghist-v1 segment quantum 一致。改变 quantum、边界或映射规则必须升级 bucketizer version，旧 tuned rows 不得用新规则重解释。

每个 immutable bucketizer artifact 至少包含：

~~~text
descriptor version
bucketizer version
op_variant_id
work_quantum_rows
padded-work integer boundaries and closed/open convention
route classifier integer rules
canonical serialization and digest
OOD result
~~~

### 6.4 动态 M_e 运行时合同

- M_e 的唯一权威来源是 EP dispatch 与 local sort 之后的 local tokens_per_expert；MUST NOT 从 pre-dispatch router map、top-k ids 或 expert mask 重新计算或二次解释（第 4.2 节边界）。
- counts/offsets 的 dtype、shape 与 device residency MUST 显式定义并进入 ABI：V1 固定 int32、device-resident；int32 operand indexing overflow 预检 fail-closed（第 15.2 节）。
- local expert id ↔ weight slot（Parameter 顺序、kernel weight bank index）的映射 MUST 显式、稳定、与 state_dict 顺序一致；MUST NOT 从 kernel 执行顺序反推 expert id。
- MAY 使用 active_expert_ids（nonempty 压缩列表）作稀疏执行优化；active 压缩不得改变 M_e 语义、ledger obligation 或 empty-expert 合同。
- P1 hot path MUST NOT 逐 expert 调用 .item()，MUST NOT 为 counts/offsets 新增 D2H 同步；任何必需的 host 侧读取只能发生在 routing epoch 构建期。
- max_M_e、ΣM_e、offset 宽度与 capacity 上限 MUST 预注册；超限是 typed failure（capacity/width violation），不得静默截断或回退 BF16。
- extreme token count 由 batch/top-k/capacity 推导：令 T 为本 microbatch 进入本 rank local sort 的 token 数（T = MBS × seq_len），k 为 router top-k，则 ΣM_e = T·k（无 drop 时），single-expert extreme M_e ≤ min(T·k, capacity_e)（capacity_e 为 dispatch capacity 生效时的 per-expert 上限；无 capacity 时取 T·k）。实现 MUST 按该上界预注册 max_M_e envelope。

---

## 7. AITER API 与 ABI

### 7.1 Ownership boundary

| Owner | Responsibilities |
|---|---|
| AITER | GPU kernels、public wrapper、compiled ABI、workspace/alignment contract、metadata builder、unit tests、microbenchmarks、tuning/config |
| Lumen | module/autograd、Parameter mapping、weight cache、optimizer invalidation、dispatch/fallback、RNG allocator、ledger、checkpoint/replay、Megatron integration |

Lumen MUST NOT 新增临时 GPU kernel，也 MUST NOT 调用 AITER private kernel body。AITER MUST NOT 依赖 Lumen 实现或测试其 operator。

### 7.2 Python facade 与 compiled ABI

Python facade MAY 使用 dataclass，例如：

~~~text
StaticPlanV1
RoutingMetadataV1
BoundExecutionV1
EphemeralOperandViewV1
~~~

V1 runtime schema ID 固定为：

~~~text
lumen.grouped_expert_runtime.v1
~~~

每个成功 bind 产生唯一 binding_id，供 launch、late attachment、ledger 和 trace 关联。

低层 custom-op ABI 只允许：

~~~text
Tensor
Tensor[]
fixed-width scalar / enum
fixed-length flat tuple
~~~

禁止 Python opaque plan handle 跨入 compiled ABI。Facade MUST 将 plan 展开为 config_id + metadata tensors + scalars。所有 output/workspace mutation 在 schema 中显式声明。每个低层 symbol 带 _v1；compiled backend 导出真实 ABI major/minor，且与 numeric/layout versions 分离。

本节与第 7.6 节的 compiled ABI 为 **draft**：A0 ABI review 签署（plan §7 A0 exit）前不冻结；头部状态 "P1 compiled ABI draft" 即指此。

### 7.3 Four-stage execution model

~~~text
query_static
  -> prepare_kernel
  -> bind_microbatch
  -> launch_*_v1
~~~

1. query_static(spec)：纯 capability；不分配、不编译、不运行 kernel。
2. prepare_kernel(spec)：加载并解析静态 key 对应的 registry，完成允许的 JIT，并生成该静态域内全部不可变 candidate plans。
3. bind_microbatch(...)：基于本次 route descriptor 做内存中的 exact-map selection，并完整事务式绑定 Fprop+DGrad+WGrad bundle。
4. launch_*_v1(...)：只执行；不得 JIT、CSV lookup、扩容、隐藏分配、异常探测或切换 backend。

本文所称 bind/launch 零 lookup，是指无 filesystem、CSV/config parse、JIT 或 autotune；bind 允许对 prepare 已构造的 immutable in-memory exact map 做确定性 O(1) route-dependent selection。

### 7.4 Transactional bind and commit point

bind_microbatch 的内部顺序固定为：

~~~text
receive already-prepared immutable candidate plans
  -> side-effect-free preflight of all allowed candidates
  -> select exactly one backend/config bundle
  -> reserve/materialize all required cache/workspace/staging
  -> validate full bundle
  -> construct immutable BoundExecutionV1
  -> return = commit point
~~~

返回 BoundExecutionV1 后，backend family、Fprop/DGrad/WGrad config、routing metadata、weight/cache epoch、runtime tensor schema、main-grad destinations、workspace slices、capacity/alignment/alias contract、prepared layouts 和 SR reservation token 均不可改变。

Candidate probing 只允许以下封闭 miss 枚举触发 pre-commit 下一个候选：

~~~text
STATIC_CAPABILITY_MISS
ROUTING_BUCKET_MISS
BACKEND_ALIGNMENT_MISS
TUNING_POLICY_MISS
CACHE_BUDGET_MISS
~~~

以下错误 MUST fail，不得换 backend：

- public ABI 输入非法；
- ABI/layout/version mismatch；
- caller workspace/capacity 不足；
- illegal alias；
- actual OOM；
- cache build/JIT error after candidate selection；
- kernel launch 后错误；
- asynchronous GPU fault。

继续执行 BF16 可能掩盖 caller bug 或损坏的 GPU context，因此上述错误不属于 fallback。Telemetry MUST 同时记录 failure_phase 与 FailureDisposition={CANDIDATE_MISS,RUN_FATAL}。

### 7.5 Late attachment of backward operands

Forward 时不要求知道未来 dY.data_ptr()。Backward 使用：

~~~text
attach_runtime_operands_v1(bound_pass, dy, dx_out, ...)
  -> EphemeralOperandViewV1
~~~

它只验证 dtype、shape、stride、device、storage offset、capacity、alignment、alias、routing/layout epoch 和 tensor/stream lifetime。它 MUST NOT 查 tuning table、分配、扩容、JIT、重新选择 backend/config、重建 weight、预留 RNG 或 fallback。

失败统一为：

~~~text
RUNTIME_OPERAND_CONTRACT_VIOLATION
~~~

若 backend 要求上游 allocator 无法保证的特殊 alignment/layout，bind MUST 预留固定 staging buffer，并把 copy/pack 纳入 config、workspace、memory accounting 和 benchmark。否则该 backend 在 commit 前就是 BACKEND_ALIGNMENT_MISS。

EphemeralOperandViewV1 只存在于 Python 生命周期层；compiled ABI 仍接收显式 tensors/scalars。

### 7.6 Low-level operator families

ABI 至少提供以下 _v1 families；精确参数由 ABI review 固定，但必须遵守前述类型限制：

~~~text
grouped_mxfp4_build_weight_cache_v1
grouped_mxfp4_quantize_row_v1
grouped_mxfp4_quantize_transpose_v1
grouped_mxfp4_fprop_v1
grouped_mxfp4_dgrad_v1
grouped_mxfp4_wgrad_v1
~~~

WGrad 的逻辑 signature 包含 mandatory Tensor[] out、output_mode 和 caller-owned workspace。API MUST 显式返回或暴露不支持原因，不允许 Lumen 捕获任意异常来猜测 fallback。

Compiled ABI（draft，A0 签署前不冻结）必须对以下字段给出精确、可测试的定义；任一缺失即 ABI 不完整，A0 不得签署：

- packed payload 与 scale 的 shape/dtype/stride/device；
- expert weights 的 bank / Tensor[] / pointer table 表达；
- counts/offsets 的 dtype、shape 与 GPU/CPU residency；
- Fprop caller-owned y_out；
- DGrad caller-owned dx_out；
- WGrad Tensor[] out / main_grad 目的地；
- workspace query / size / alignment；
- output dtype enum；
- alias/mutation rules；
- stream/lifetime；
- all-empty 返回语义；
- typed capability/status/error。

### 7.7 Backend family、execution mode 与 no-CK 契约

**Sequential (P0a/P0b)** 是正规 correctness backend：Lumen 用 Megatron post-dispatch/local-sort 的 x_sorted + tokens_per_expert 切片、稳定映射原 per-expert Parameters，复用 dense MXFP4 Fprop/DGrad/WGrad；不以 grouped MoE backward 先交付为前提。Megatron 负责 routing/dispatch/combine/permutation；Lumen 仍必须完成 §11.4 autograd/context、FC1–SwiGLU/clamp–probability–FC2 衔接及 router probability gradient、FP32 main_grad exactly-once、empty/ragged/recompute/cache/optimizer 生命周期，不能把 permutation 已正确当作完整训练闭环。

**Grouped (P1)** 是逐项复用/扩展已有 AITER 能力的 MXFP4 F/D/W 训练 bundle。Fprop/DGrad 是 M-grouped，WGrad 是 K/reduction-grouped；消费与 sequential 相同 recipe、BF16 snapshots、canonical payload/scales、H16 与 RNG logical mapping（prepared repack 必须值保持）。统一调度可使用少量 launches，但不承诺只有一个 kernel 或所有 routing 下总更快。同一 wrapper SHOULD 使切换模式不改变 router/model/state_dict、数据和 mask 语义。Forward bind 固定整个 F/D/W bundle 与 execution_mode；backward 只能 late-attach runtime operands，不能重新挑选模式/backend。

execution_mode 与 kernel backend family 是两个独立维度，命名不得混用：

~~~text
execution_mode ∈ {grouped, sequential}   expert 维度的执行结构
kernel_family  ∈ {FLYDSL, TRITON, ASM}   底层 kernel 实现族
~~~

本 feature 的 quantization、cache preparation、Fprop、DGrad 和 WGrad MUST NOT 选择 CK（Composable Kernel），包括任何内部 auto-dispatch 路径。kernel_family 的 V1 allowlist 为 {FLYDSL, TRITON, ASM}；Gluon 不在 allowlist（用户本轮计划明确只允许上述三类）。任一 candidate plan 的 kernel_family MUST 在 capability 与 BoundExecution 中显式记录；manifest、ledger 与每个 kernel repr 同步记录 kernel_family。

Strict gate 每 rank 断言 ck_invocations=0；任何 CK 命中均使 strict run 失败。P0 复用的现有 Dense MXFP4 调用链同样受本契约审计，不得因走 dense 既有路径而豁免。

---

## 8. Backend selection、strict 与 fallback

### 8.1 CLI

以下为待实现 CLI 契约，不表示当前代码已可用。保留既有唯一选择开关 `--lumen-moe-mxfp4-backend`：其 backend 值表达 MoE execution_mode（sequential/grouped），不是 kernel_family；不得再造同功能 execution-mode 开关。auto 是既有候选选择策略，不是第三种执行结构。新增且只新增以下主开关：

~~~text
--lumen-moe-mxfp4
--lumen-moe-mxfp4-backend {auto,grouped,sequential}
--lumen-moe-mxfp4-strict
--lumen-moe-mxfp4-grouped-tuning-policy \
  {require-certified-bucket,allow-validated-heuristic}
~~~

沿用既有建议默认（仍是待实现设计默认，未核实为运行时默认；本轮不改变默认值或引入新 auto 策略）：

~~~text
backend=auto
strict=false
grouped-tuning-policy=require-certified-bucket
~~~

scope 固定为 routed-expert FC1/FC2；不得使用含义过宽的 --linear-fp4，也不新增同功能环境变量。若与 LUMEN_DSV4_LINEAR_FP8 同开，启动 MUST 失败。

正式 P0/P1/promotion gate MUST 显式指定 backend；不得使用 auto。exact_continue V1 也 MUST 使用显式 backend/config。

### 8.2 State machine

| Requested backend | Strict | Behavior |
|---|---:|---|
| sequential | true | only sequential MXFP4 |
| grouped | true | only grouped MXFP4 |
| auto | true | pre-commit select grouped or sequential MXFP4; never BF16 |
| sequential | false | sequential MXFP4 -> BF16 |
| grouped | false | grouped MXFP4 -> sequential MXFP4 -> BF16 |
| auto | false | select grouped/sequential MXFP4; BF16 only if both unavailable |

对 auto，grouped capability/bucket miss 后选 sequential 是 selection，不是 precision fallback。对显式 grouped，降到 sequential 是 execution-mode fallback（既有 backend_fallback counter 记录此事件）；底层 kernel_family 与 execution_mode 分开。任一路径 MXFP4 降到 BF16 是 precision fallback。requested_backend=sequential 的主动 correctness 路径不是 fallback。Production promotion 的 P1 candidate 在任何 rank、FC/pass 都禁止 grouped→sequential 或→BF16；此必要条件不会把 V1 的 experimental 声明上限升级为 production-ready。

require-certified-bucket 只过滤 grouped candidate：

- grouped + strict 的 row miss：pre-launch fail；
- auto + strict 的 row miss：可选 sequential MXFP4，禁止 BF16；
- explicit sequential：不受 grouped tuning policy 影响；
- non-strict：所有 MXFP4 candidate 均不可用后才允许 BF16。

计数真值（每 FC/pass/rank）：保留累计 pre-commit selector 决策事件口径，而非已运行 backend 的转换次数。`backend_fallback` 是显式 grouped 请求在允许 miss 后降到 sequential 候选的 execution-mode 降级事件，不表示曾执行 sequential；`precision_fallback` 是允许 miss 后进入 BF16 候选的精度降级事件，也不表示先执行过 MXFP4。普通 candidate attempt/miss 本身不递增这两个计数，只有按既定链发生的降级决策递增。

| Pre-commit decision chain | Final executed（成功非 EMPTY） | selector mode-degradation events (`backend_fallback`) | selector precision-degradation events (`precision_fallback`) |
|---|---|---:|---:|
| explicit sequential → sequential MXFP4 bundle | sequential MXFP4 | 0 | 0 |
| explicit grouped → grouped MXFP4 bundle | grouped MXFP4 | 0 | 0 |
| explicit grouped → sequential MXFP4 candidate（仅 non-strict、允许 miss） | sequential MXFP4 | 1 | 0 |
| explicit sequential → BF16 candidate（仅 non-strict、sequential miss） | BF16 | 0 | 1 |
| explicit grouped → sequential MXFP4 candidate → BF16 candidate（仅 non-strict、两者 miss） | BF16；未执行 sequential/MXFP4 | 1 | 1 |
| auto → grouped/sequential MXFP4 candidate | selected grouped/sequential MXFP4 | 0（selection 单列） | 0 |
| auto → BF16 candidate（两个 MXFP4 candidates 均不可用、non-strict） | BF16；未执行 MXFP4 | 0（selection 单列） | 1 |

Explicit grouped 的 BF16 结果必须沿既定 grouped→sequential→BF16 候选链：不新增跳过 sequential 的 grouped→BF16 路径。事件记录 MUST 含 from/to mode 与 precision、reason、failure_phase、commit_state=PRECOMMIT，并与 candidate_attempt/miss 记录单列；即使最终 BF16，前面的 selector mode 降级事件仍保留。Execution 字段只记录最终真实执行的 bundle/mode/kernel_family/precision，绝不把候选访问写成 kernel execution。EMPTY 按 §6.2 记录无 GEMM（必要 zero-fill 单列），失败/未 launch 记录相应 terminal state，不捏造 Final executed 成功。

Requested/selected/executed execution_mode、kernel_family、executed_precision 与两个计数 MUST 分开记录；既有 requested_backend 字段是 CLI selector alias，不是 kernel family。Strict sequential 禁止 BF16；strict grouped 禁止 sequential 和 BF16。所有箭头仅表示 §7.4 封闭 candidate miss 下的 pre-commit 候选决策，不允许运行后尝试降级；selected 最终 bundle 与 executed 必须一致，否则 run-fatal。非 strict 也不允许 public ABI/version/alias/capacity/OOM/launch/async 等 fatal errors 触发 fallback。

### 8.3 Fallback timing

Fallback/selection MUST 在第一个 mutation 或 kernel launch 前完成。BoundExecutionV1 commit 后不得 fallback。用户主动请求 sequential、auto selection、backend fallback 和 precision fallback 必须是不同 telemetry 状态。

---

## 9. Weight epoch、cache 与 memory budget

### 9.1 Authoritative cache identity

Cache key 至少包含：

~~~text
parameter identity
committed_weight_epoch
cache_generation
training_recipe_id
operand_quant_recipe_id
operand_transform_id
physical_layout_id
prepared_layout_id
device / gfx / CU
shape / local-E
~~~

committed_weight_epoch 是权威数值提交版本。Parameter._version、identity 和 checksum 只作额外防御，不能替代它。

推进顺序：

~~~text
FP32 optimizer update succeeds
  -> FP32 master syncs to BF16 Parameter
  -> readiness event completes
  -> atomically advance committed_weight_epoch
~~~

overflow/skipped step 不推进。一个 optimizer step 内所有 microbatches 和 recompute 使用同一 weight epoch。

### 9.2 Invalidation

- optimizer step、checkpoint load、支持的参数原位提交、device/dtype move、reshard 均使 derived cache 失效。
- mark_mxfp4_weights_dirty() 表示一次受支持的外部权重提交：只允许在无 accumulation window、无异步 consumer 的安全边界调用；它推进 weight epoch、推进 cache generation 并清除 derived cache。
- checkpoint load 恢复 checkpoint 中的 weight epoch，同时推进 process-local cache_generation 并清空 derived cache。
- 无法拦截的裸 .data 修改明确为 unsupported。

### 9.3 Cache content and generations

每个 expert 从同一 BF16 snapshot 构造 forward canonical layout 与 DGrad transpose view。Production path 只允许：

- 一代 active cache generation；
- 一个 selected prepared layout；
- 不按 tuning config 重复缓存完整权重；
- old consumer completion 后优先释放 retired generation，再构建新 generation。

Parity test 需要双 backend cache 时 MUST 使用显式 test override，并单独报告内存。

### 9.4 Budget

定义 B_ref 为本 rank、本 device、启用 scope 内所有 local routed-expert FC1/FC2 BF16 source-weight 的 unique-storage bytes，不含 shared expert 与 optimizer state。

~~~text
default persistent cache budget = 0.75 x B_ref
maximum persistent budget        = 1.00 x B_ref
default peak-build budget        = 1.00 x B_ref
absolute maximum                 = 1.00 x B_ref
~~~

以下五类 MUST 分别计量、分别对照限制报告，不得合并为单一内存数字：

~~~text
persistent_cache_bytes         常驻 derived cache（payload/scales/transpose/metadata）
peak_cache_build_bytes         构建期 owned derived storage 高水位
reusable_workspace_bytes       预分配、跨 launch 复用的 workspace/staging
transient_launch_bytes         单次 launch 的瞬时 staging/copy
device_total_preflight_bytes   preflight 时本 rank 本 device 的总需求
~~~

persistent_cache_bytes 对照 0.75x 默认 / 1.0x 硬上限；peak_cache_build_bytes 对照 1.0x B_ref；reusable_workspace_bytes 与 transient_launch_bytes 必须列入 planner 预测与 manifest；device_total_preflight_bytes 另对照实测 free HBM 做 fail-fast。

Test-only cache-disabled P0 oracle：P0 必须提供一条不经 weight cache、每次调用从当前 BF16 Parameter 现算量化的 test-only 路径；cache-enabled 与 cache-disabled 的 packed bytes、scale bytes、layout IDs 与 GEMM 输出 MUST bitwise 一致（cache parity gate）。

预测为 0.82x 时，默认 MUST 返回 CACHE_BUDGET_MISS。只有用户显式配置且 manifest 记录 provenance，才能把 persistent budget 提高到 0.82–1.0x。

两层 enforcement：

1. bind 前 planner 计算 predicted_persistent_bytes 和 predicted_peak_build_bytes；
2. cache allocator 每次 allocation 前原子 reserve 并执行硬配额。

实际计费包括 payload、scales、transpose/swizzle、metadata/descriptors、offsets、alignment padding、persistent auxiliaries、preallocated workspace/staging、builder scratch 和尚未释放的 retired generation。Shared/aliased storage 只计一次。predicted_peak_build_bytes 是构建期间所有 owned derived storage 的高水位。

Promotion eligibility 按本 rank 全部 local experts 的 steady-state footprint 计算，不因当前 microbatch 只命中少数 experts而缩小。当前 free HBM 只可作为额外 fail-fast，不能自动改变 manifest 中的预算或 backend selection。

计划超限是 pre-bind candidate miss；allocator 超过已批准 plan 是 ALLOCATOR_PLAN_BREACH 且 run-fatal。若旧 generation 未退休导致 peak 超限，只能等待安全退休后重建或 candidate miss，不能越限。

---

## 10. SR RNG contract

### 10.1 Ownership and domains

SR 使用 Lumen-owned、checkpointed Philox counter allocator。禁止 Python random、Torch global RNG 或纯 identity hash。

V1 RNG schema ID 固定为：

~~~text
lumen.philox_logical_draw.v1
~~~

每个 global_rank、rng_domain 使用独立 seed：

~~~text
domain_seed = stable_derive(
  run_seed,
  global_rank,
  rng_domain_id,
  rng_schema_version,
)
~~~

建议至少拆分 row DGrad dY 与 transposed WGrad dY domains；domain 列表与 ID 属于版本化 schema。

### 10.2 Deterministic allocation

1. 在 microbatch obligation declaration 阶段收集全部 SR obligations。
2. 按版本化 canonical logical key 排序。
3. 对每个 domain 一次性、原子预留 ranges。
4. Bind 只取得已经分配的 token。
5. Candidate probing 不消耗 draw。
6. recompute、physical retry 和相同 logical invocation 复用同一 token。

Canonical key MUST 来自训练图逻辑身份，例如 optimizer step、microbatch、global layer、FC role、pass、operand role 和 logical invocation ID；MUST NOT 包含 backend、config、tile、pointer 或 physical attempt。

Token 至少包含：

~~~text
rng_schema_version
rng_domain_id
logical_invocation_id
philox_seed
base_draw
draws_reserved
draws_used
~~~

映射固定为：

~~~text
absolute_draw  = base_draw + logical_element_index
philox_counter = absolute_draw // 4
philox_lane    = absolute_draw % 4
~~~

logical_element_index 由 operand layout spec 定义，不由 tile、wave、backend/config 或 launch order 定义。Row stream 按 sorted tensor row-major；transposed stream 按 expert-major、feature-major、padded-token-major 展平。

### 10.3 Reservation rules

- padded lanes 占 draw slot；
- empty expert 消耗零 draw，但仍有 ledger obligation；
- row dY 与 H16-transposed dY 使用不重叠 subrange；
- fallback/abort 后未使用的已预留 range不回收；
- A4/A8 fixed-snapshot test 的对应元素共享 draw；
- reservation 前检查 base_draw + draws_reserved 的 uint64 overflow；
- checkpoint 保存每个 domain 的 seed、next_draw 和 next logical invocation ID；
- AITER 只消费 SR token 并回报 draws_used，不生成 seed、不推进全局 counter。

---

## 11. Autograd、WGrad 与 lifetime

### 11.1 Saved state

Forward 只保存 compact FP4 operands、immutable routing metadata、bound execution、weight/cache generation、SR tokens 和必要 events。不得额外保存完整 BF16 weight bank。

Autograd context MUST 强引用上述对象直到所有异步 Fprop/DGrad/WGrad consumer 的 completion event 完成。

### 11.2 Eager WGrad only

V1 只支持 backward 中立即提交 WGrad，并直接写第 5.5 节定义的 FP32 main_grad。若用户启用 deferred/delay WGrad，启动 MUST 失败，不能静默改成 eager。

Deferred WGrad 放入 P2；未来实现必须重新定义 typed record、bounded queue、backpressure、exactly-once、RNG/lifetime 和 checkpoint drain barrier。

### 11.3 Recompute

Activation recompute 为同一 logical invocation 的 physical re-execution：

- 使用同一 committed weight epoch；
- 使用同一 routing metadata identity；
- 使用同一 RNG token；
- ledger 新增 physical attempt_id，不新增 logical obligation；
- 不允许 recompute 触发新 backend selection 或新 cache generation。

### 11.4 Lumen framework adapter contract

V1 的框架侧实现 MUST 固定以下契约，并作为独立 review 对象：

1. **Module 接口：** 与 Megatron GroupedMLP 兼容的 forward 签名；输入为 expert-sorted x_sorted 与 post-dispatch、post-local-sort 的 tokens_per_expert；per-expert weight 保持 Parameter 语义与 weight0..weightN state_dict keys；每个 expert 的 recipe/cache 独立。
2. **Custom autograd：** forward 只保存 compact packed operands、immutable routing context 与 bound execution（第 11.1 节）；backward 调 DGrad + WGrad。
3. **WGrad exactly-once：** WGrad 对本 invocation 恰好一次直接累加 FP32 main_grad（ACCUMULATE_FP32）。当 WGrad 已直接写 main_grad 时，weight 输入的 backward 返回值 MUST 为 None；MUST NOT 经 autograd 再生成一份 Parameter.grad 造成二次累加。
4. **main_grad 分配时机：** 二选一并冻结——(a) forward bind 前保证已分配，或 (b) 受限的 backward late-attachment 路径在 attach 时验证并绑定；任一 backend MUST NOT 在 kernel launch 后才发现 main_grad 不存在。
5. **Megatron gradient bookkeeping：** main_grad 的设置/清除归属 MUST 显式：V1 固定为 Lumen adapter 在 optimizer zero-boundary 侧登记，Megatron 既有 gradient bookkeeping 语义不变更；MXFP4 路径不得绕过 Megatron 的 grad buffer 归属。
6. **Accumulation window 归属：** window 的 open/close 由 optimizer step 语义拥有；WGrad 只在已声明的 open window 内 accumulate，window 外 attach MUST fail。
7. **Reducer/all-reduce hook 完成信号：** dW 完成的判定来源固定为 WGrad stream completion event + ledger obligation COMPLETED；reducer hook 只在二者闭合后消费 main_grad。
8. **Parameter 顺序与 local expert id 映射：** Parameter[i] ↔ local expert id 的映射 MUST 显式单向且与 state_dict keys 一致；MUST NOT 从 kernel 执行顺序反推 expert id。

---

## 12. Ledger 与可观测性

### 12.1 Obligation state

每个 FC1/FC2、Fprop/DGrad/WGrad logical obligation 在 backend selection 前声明：

~~~text
DECLARED
  -> BOUND
  -> ENQUEUED
  -> COMPLETED | EMPTY | FAILED_PRELAUNCH | ABORTED
~~~

COMPLETED 只能在 stream event 确认后落账。Sequential per-expert launch、split-K 或 staging copy 记录为 child kernel_sublaunches，不增加 logical obligation。

守恒式：

~~~text
declared = completed + empty + failed_prelaunch + aborted + open
~~~

正式 grouped strict success 必须满足：

~~~text
open = 0
failed_prelaunch = 0
aborted = 0
backend_fallback = 0
precision_fallback = 0
ck_invocations = 0
all completed obligations executed grouped MXFP4
~~~

### 12.2 Independent attributes

以下是独立属性，不是 mutually-exclusive terminal states：

~~~text
requested backend
selected backend
selection/fallback chain
executed precision
precision fallback
config source
route descriptor/bucket
recipe/layout IDs
kernel_family
SR token/draws
logical_invocation_id / physical attempt_id
~~~

Expected obligations MUST 从训练图/autograd context 生成，不能从 kernel 次数反推。全 rank 校验使用聚合 counts + digest，失败时保存 rank detail。Strict failure 在安全同步点全局传播，避免 collective hang。

### 12.3 Compatibility、strict 与 certification 分轴

三个独立轴，不得合并为单一布尔：

~~~text
runtime_compatible   ABI/layout/capability/workspace 可运行（第 14.3 节 predicate）
strict               是否允许 backend/precision fallback（第 8 节状态机）
tuning_policy        certified bucket 或 validated heuristic（第 8.1、14 节）
~~~

certification 是更进一步的独立属性，至少拆分：

~~~text
correctness_validated
tuning_config_certified
artifact_provenance_certified
~~~

promotion 与 exact_continue 在 runtime_compatible 之上额外要求 certified。uncertified 不自动等于 runtime incompatible：runtime_compatible 只回答"能否运行"，certification 回答"证据是否被认证"；二者分离使 allow-validated-heuristic 可用。strict 只描述 precision/fallback 语义，不等价于任一 certification 状态。

### 12.4 Required telemetry

每次 logical invocation 至少记录：

~~~text
layer / FC role / pass
binding ID / runtime schema
requested_backend / selected execution_mode
actual execution_mode / kernel_family / executed precision
selection and execution-mode fallback chain / precision fallback
E_l / M / N / K
route descriptor version and fields
bucketizer version / padded-work bucket / route bucket
training/operand-quant/operand-transform recipe IDs
physical / prepared layout IDs
kernel_family
config ID / config source
weight epoch / cache generation / routing epoch
SR token and draws receipt
failure phase / reason / disposition
kernel sublaunch count
~~~

首次出现某个 layer/role/reason 时 warning，后续只计数。所有 ranks 汇总 counts + digest；rank 0 只负责展示，不能只观察本 rank。

---

## 13. Checkpoint and continuation modes

### 13.1 warm_start

- 只加载 BF16 model weights；
- optimizer、scheduler/scaler、RNG、data cursor、weight epoch 重新初始化；
- derived cache 不加载。

### 13.2 exact_continue

只允许同 topology、rank mapping、build manifest、kernel/config、stream schedule、optimizer、data pipeline 和 deterministic settings。V1 禁止 backend=auto、未冻结的 config mapping、validated heuristic、fallback、resume 后 autotune/heuristic/alternative-candidate selection 和未认证的 config selection。

Replay certificate MUST 固定显式 backend，以及从完整 static key + exact route bucket key 到 config_id/prepared_layout_id 的 immutable versioned mapping 和 artifact digest。Fresh resume process MAY 从同一认证 artifact 确定性重建 StaticPlan/JIT code，但 MUST 验证 mapping 和 kernel digest 一致；不得新增、删除或替换 mapping row。

Checkpoint load 会清空 derived cache，因此 resume 后第一个 microbatch MUST 使用该冻结 mapping 与重新构建的 cache 执行一次正常 transactional bind，生成新的 BoundExecutionV1；禁止复用 checkpoint 前的 BoundExecution，也禁止把 prepare/bind 当作重新选型机会。

replay_class 分开表示：

~~~text
BITWISE_REPLAY_CERTIFIED
STATE_COMPLETE_ONLY
REPLAY_UNSUPPORTED
~~~

数值稳定性/质量是独立字段。只有完整 run-level replay certificate 为 BITWISE_REPLAY_CERTIFIED 时，模式才可命名 exact_continue；否则必须称 state_complete_continue。

STATE_COMPLETE_ONLY 不是 exact 失败后的自动降级标签。它必须通过第 16.2 节的独立 state-completeness qualification；restore-state、cache rebuild 或 continuation-stability 任一门禁失败时，replay_class 必须为 REPLAY_UNSUPPORTED。该分类只证明 checkpoint 状态完整且可继续训练，不证明逐 step replay、数值等价或质量继承。

Replay certificate 绑定：

- GPU/driver/ROCm/AITER/RCCL；
- topology/rank mapping；
- 所有 pass config/kernel digest；
- routing/shape envelope；
- split-K/reduction order；
- stream schedule；
- optimizer/data pipeline；
- RNG contract；
- execution manifest digest。

BITWISE_REPLAY_CERTIFIED 还要求无不确定 atomic accumulation；split-K 必须为单 partition，或使用 certificate 固定的确定 reduction order。任一参与 pass/rank 不具备该能力时，启动或 bind fail closed。

Checkpoint 只能在全 rank quiescent optimizer-step boundary 保存，且 open ledger entries=0。它恢复 model、FP32 master、optimizer、scheduler/scaler、data cursor、Python/NumPy/Torch CPU/CUDA/model-parallel/data-worker RNG、FP4 counter-RNG、committed weight epoch 和 compact ledger frontier；derived cache load 后无条件清空。

### 13.3 Hash-chain frontier

Checkpoint 不保存完整 invocation history。V1 固定可追加 hash-chain：

~~~text
ledger_schema_version
ledger_epoch
next_logical_invocation_id per domain
RNG seed / next_draw per domain
cumulative terminal counts
prefix_leaf_count
hash_chain_state
last committed optimizer step
execution/certificate manifest digest
~~~

V1 replay schema ID 固定为：

~~~text
lumen.replay_hash_chain.v1
~~~

Hash chain 定义为：

~~~text
H0 = SHA256(replay_schema_version || frozen_run_manifest_digest)
H(i+1) = SHA256(H(i) || canonical_event_v1)
~~~

canonical_event_v1 至少包含 logical key、stable logical-binding key、config ID、routing/weight epoch、完整 SR token、shape/dtype/layout、terminal status 和 canonical attempt digest。每次 bind 唯一的 process-local `binding_id`、地址、时间戳和 branch/process UUID 只用于 trace correlation，MUST NOT 进入 replay hash。canonical attempt digest 只包含有序的语义结果和 sublaunch descriptor，不包含 ephemeral identity。

Same-manifest resume 在首个 op 前验证 frontier；缺失、重复、乱序或 manifest mismatch 均为 REPLAY_FRONTIER_MISMATCH。Checkpoint commit 时 uninterrupted branch 也必须关闭当前 segment；uninterrupted 与所有 resume branches 从同一 checkpoint frontier 创建相同 deterministic logical child segment，并记录 parent_frontier_digest。外部 artifact 可另记唯一 physical attempt/branch ID，但该 ID 不进入 canonical hash。完整 invocation 明细保存为外部 artifact。

### 13.4 reshard_continue

仅用于 Megatron/DCP 的 EP/DP/TP/PP/world-size 改变；V1 不含 FSDP。它保证 canonical global Parameter/optimizer mapping、scheduler、step 和 data cursor 正确并可继续训练，不承诺 bitwise replay。

每个受支持 source-to-target tuple MUST 在 manifest 中显式列出并单独验收；单向测试不授权反向或任意拓扑声明。

Reshard 不使用 same-manifest frontier equality。它必须定义 versioned `reshard_transition_id`，验证 source frontier/source manifest，记录 target manifest，并以如下 canonical transition 开启新的 ledger epoch：

~~~text
target_H0 = SHA256(
    replay_schema_version
    || source_frontier_digest
    || source_manifest_digest
    || target_manifest_digest
    || reshard_transition_id
)
~~~

RNG transition 必须按 domain 显式定义，不能比较不同 rank ownership 下的 raw per-rank state：

- 与 global sample/parameter identity 绑定且可 canonical remap 的 domain，按注册映射恢复；
- data-worker/model-parallel domain 从 global data cursor、target rank mapping 与 transition ID 确定性重建；
- rank-local FP4 SR domain 使用 `stable_derive(run_seed, source_frontier_digest, target_manifest_digest, target_global_rank, rng_domain_id, rng_schema_version)` 生成新的 target epoch seed，`next_draw=0`、`next_logical_invocation_id=0`；二者由新的 target ledger/RNG epoch 命名空间隔离，不与 source token identity 混用；
- source RNG state 与 target mapping/seed ledger 均保存于 artifact，禁止隐式复用可能重叠的 draw range。

---

## 14. Tuning、bucketization 与 certification

### 14.1 Grouped tuning key

通用 static key 只包含 route-independent fields：

~~~text
gfx / CU
pass
op_variant_id
E_l / N / K
training_recipe_id
operand_quant_recipe_id
operand_transform_id
physical_layout_id / prepared_layout_id
output mode / dtype
~~~

fc_role 不进入通用 key；N/K/pass 已表达差异。只有 epilogue 等真实语义差异才使用 op_variant_id。

Exact row lookup 对象是：

~~~text
static key
+ descriptor version
+ bucketizer version
+ padded-work bucket
+ route bucket id
~~~

不是 raw group_sizes，也禁止 route 字段 wildcard。require-certified-bucket 禁止 nearest-bucket、跨 CU relaxation 或缺列 wildcard。

### 14.2 Bucketizer evidence

Bucketizer 由真实 routing artifacts 训练/审计，tuning artifacts、validation artifacts 与 final held-out certification artifacts MUST 按完整 capture run/seed/data-shard/checkpoint 分离。一旦 final set 被用于修改 bucket、候选或阈值，它就不再是 held-out。

Coverage 至少包含：

- bucket center 与两侧边界；
- size 两端；
- maximum padding waste；
- single-hot；
- many-empty/small-active；
- balanced；
- extreme skew/long tail；
- 多个真实 layer、step 和 seed。

Synthetic artifacts 只补 adversarial corner，不能单独获得 provenance certification。每个待认证 exact key/bucket 至少有三个真实 artifacts：center、boundary、extreme，并来自至少两个独立 capture runs。

V1 使用 route-representatives-v1 选择代表：

- center：在 bucket 内，以各 descriptor 数值字段的 robust range 归一化后，选择到其他 artifacts 的 L1 距离和最小的 medoid；
- boundary：选择到任一 bucket integer decision boundary 的最小 normalized slack；
- extreme：选择 hot_share、empty_share、padding_waste、max_rows 四项 empirical rank 最大值最高的 artifact；
- 所有 tie 按 canonical artifact digest 的字节序打破。

其中：

~~~text
hot_share     = max_rows / max(1,total_rows)
empty_share   = (E-nonempty) / E
padding_waste = (padded_rows32-total_rows) / max(1,padded_rows32)
~~~

Certification envelope 为每个字段的闭区间 min/max：

~~~text
total_rows, nonempty, max_rows, padded_rows32, tiles32,
hist16[0] ... hist16[7]
~~~

Runtime descriptor 必须逐字段落在 envelope 内，并满足 config 的 max_rows_capacity。

对真实 artifact r 和候选 config c，定义：

~~~text
L(c,r) = fresh-process paired measurements 的 median latency
oracle(r) = min L(c,r) over all correctness-valid, runtime-compatible candidates
regret(c,r) = (L(c,r)-oracle(r)) / oracle(r)
weighted_mean_regret = sum(freq_r * regret(c,r)) / sum(freq_r)
~~~

freq_r 是 final held-out trace 中该 exact invocation 的出现次数，不再按 token 数二次加权。

V1 预注册 regret gate：

- selected config 相对 held-out candidate oracle 的 frequency-weighted mean latency regret，U95 <=3%；
- center/boundary/extreme 任一 artifact 的 regret U95 <=5%；
- 差异在 2% 内视为 tie，选择更通用、资源占用更稳定的 config。

若最优 config 翻转或超过 gate，MUST 拆桶或使用明确的 generic config；不得事后放宽阈值。最多 12 个 bucket 只是目标，不是强制压缩。

U95 使用 process-first hierarchical paired bootstrap：先按独立 capture group/process 重采样，再在 process 内按预注册 contiguous block 重采样。单个 artifact 的重复 kernel iterations 不是独立 provenance samples。

### 14.3 Runtime compatibility

每次 bind 动态重算：

~~~text
compiled ABI major matches and minor is in the declared compatible range
backend build/version satisfies the published compatibility predicate
numeric recipe / transform / physical layout / prepared layout IDs are supported
required capability bits are present
descriptor/version available
exact bucket row exists
descriptor lies inside certification envelope
max_rows <= config capacity
GPU arch/CU/ROCm conditions and static key match
workspace and alignment are valid
~~~

OOD 包括 ABI/backend compatibility miss、missing capability bit、unknown descriptor version、missing bucket、envelope miss、capacity exceeded 和 numeric/layout/arch/CU/ROCm mismatch。Strict grouped 失败；auto 可在 commit 前选择 sequential。不得 nearest-match。uncertified config 不是 OOD/runtime incompatibility：它由 tuning_policy 轴处理（require-certified-bucket 时 pre-launch miss，allow-validated-heuristic 时可用）。源码路径、git commit 和 artifact provenance 不参与 API 类型兼容判断；它们由独立 certification flags 管理。

### 14.4 Config source and representation

~~~text
config_source=certified_bucket
config_source=validated_heuristic
config_source=sequential_reference
~~~

正常 execution 不得在 launch 时 autotune。AITER runtime mapping 可使用专用 config table，并以 immutable artifact-set sidecar 记录 tuning/validation/held-out provenance。

每个 launchable Triton/Gluon kernel MUST 使用 AITER make_kernel_repr；其他 backend 使用其真实 naming/registration mechanism。Trace 至少区分 weight/cache build、row quant、transpose/H16 quant、FC1/FC2 Fprop/DGrad/WGrad 和 staging copy，并包含 op variant、route/bucket versions、config ID 和 layout IDs。

V1 只承诺满足 frozen plan 条件的 low-level launch 可以被 capture。Capture contract 必须绑定 immutable RoutingMetadata object identity、counts/offsets、每个 layout_id 的 padded offsets、operand capacities、caller-owned workspace、所有参与 tensor 的稳定地址、numeric/physical/prepared layout IDs，以及 SR token 的 draws_reserved/draws_used。routing_fingerprint 只用于诊断，不能替代 metadata identity。

若 capture 中存在 SR，compiled ABI 必须接收显式 counter/state，并由 caller 为每次 replay 提供唯一、不重叠的 reservation；kernel/backend/config 不得改变 logical-element-to-draw mapping。任何一个绑定条件或 RNG uniqueness 无法保证时，query_static MUST 返回 capture_safe=false。Dynamic-routing end-to-end replay 和 bucket-capacity graph 留给 P2。

---

## 15. Correctness test gates

### 15.1 Validation order

固定顺序：

~~~text
numeric golden vectors
  -> AITER public-wrapper kernel tests
  -> P0 sequential reference tests
  -> P1 grouped-vs-P0 parity
  -> Lumen dispatch/autograd/cache/checkpoint tests
  -> Megatron EP integration
  -> short training
  -> performance
~~~

不得用 end-to-end loss 掩盖 kernel mismatch，也不得在 correctness 未通过前调优性能。

### 15.2 Numeric/layout tests

- selector、E2M1、E4M3FN、E8M0、special values 和 saturation 使用 byte-exact golden vectors；
- row/transpose layouts、nibble order、scale placement、H16、padding 和 offsets 使用 byte-exact fixtures；
- same BF16 snapshot 的 dual layout 独立量化；测试必须能检测 decode/requant；
- identical RNG token 必须 bitwise 重放 packed gradient operands；
- backend/config/tile 变化不得改变 logical RNG mapping；
- A4/A8 fixed-snapshot 的 scale bytes 和 W4 bytes 必须 bitwise identical；
- uint64 draw overflow、int32 operand indexing overflow、illegal alias、capacity/alignment/version mismatch fail closed；
- output/workspace 周围使用 canary 检测 OOB。

### 15.3 Shape/routing matrix

至少覆盖：

~~~text
E_l in {32,64}
M_e in {0,1,15,16,31,32,33,63,64,127,128,129,257}
balanced / half-empty / many-empty / single-hot / Zipf-long-tail / all-empty
FC1 and FC2
Fprop / DGrad / WGrad
OVERWRITE BF16 / OVERWRITE FP32 / ACCUMULATE_FP32
cold/hot cache
multiple gradient-accumulation microbatches
full recompute
~~~

专门的 segment-isolation case：M=[31,1,33,...]，用 sentinel 验证 scale、pad、H16 和 RNG 不跨 expert。

还必须覆盖 noncontiguous、storage offset、wrong device/dtype、alias、stale routing/weight epoch、workspace/capacity/alignment 错误；这些均应 prelaunch fail。

### 15.4 Numerical gates

使用三个层级：

1. **Operand parity:** grouped quantizer 的 packed bytes、scales、offsets 和 metadata 与 trusted per-expert quantizer bitwise 相同。
2. **Kernel parity oracle:** 固定 packed bytes/scales/metadata，反量化后逐 expert FP32 matmul；隔离 layout/index/reduction 错误。
3. **Recipe oracle:** 从原始 BF16 X/W/dY 做独立 BF16 per-expert 数学实现；衡量 A4W4 recipe 误差。

Bring-up hard floors：

| Comparison | Fprop | DGrad | WGrad |
|---|---:|---:|---:|
| kernel parity oracle | >=30 dB | >=30 dB | >=30 dB |
| BF16 recipe oracle | >=12 dB | >=12 dB | >=10 dB |

Promotion 前必须用独立、真实 DSV4 activation/gradient pilot 预注册 production envelope。默认目标是 kernel-parity SNR >=40 dB，且 grouped 相对同 packed operands 的 P0 sequential SNR 下降 <=0.25 dB；若硬件累加顺序无法满足 40 dB，替代阈值必须由 pilot 在正式 P1 数据可见前冻结，且不得低于 bring-up floor。

Recipe accuracy 在真实 DSV4 snapshots 上相对 P0 同 seed/op 的下降不得超过 0.5 dB。Kernel parity 还要求至少 99% 元素满足预注册 atol=0.5、rtol=0.02；FP32 WGrad 另外报告 norm error 与 max error。

### 15.5 Lumen tests

必须覆盖：

- CLI scope/conflict/defaults；
- 第 8.2 节 backend/strict 六格状态机及计数真值：显式 sequential 选择零 fallback；grouped 不可用 strict typed fail；non-strict 仅允许 §7.4 封闭 pre-commit miss 降级并真实计数；fatal 与 post-commit 错误不得降级；
- grouped tuning policy 只过滤 grouped；
- typed pre-commit miss 与 non-fallback errors；
- bind commit 后禁止 selection/fallback；
- bind/launch steady path 零 filesystem/config parse、零 JIT、零 hidden allocation；仅允许已准备 exact map 的内存查询；
- late attachment validation；
- selected backend/pass/config counters；
- committed_weight_epoch、skipped update、external dirty、DCP in-place load；
- cache budget planner 与 allocator hard quota；
- gradient accumulation、full recompute、routing-probability gradient；
- eager WGrad only；deferred flag 启动失败；
- warm/exact/state-complete/reshard modes；
- ledger conservation、events 和 all-rank digest；
- low-level launch 的 capture-safe capability；dynamic-routing replay 明确为 unsupported；
- kernel_family 记录与每 rank ck_invocations=0 断言（含 P0 dense 调用链）；
- cache-disabled test-only 路径与 cache-enabled 的 bitwise parity。

### 15.6 AITER artifact requirements

每个新/修改 kernel 必须有 public wrapper、正确目录、repr、unit tests、benchmark 和 tuning config。不得复制近似 kernel；共享 activation/shuffle/reduction/helper 放 AITER utils/common。测试调用 public wrapper，不允许 literal tuning dict 绕过真实 config resolution。

通过本节只允许声明 tested gfx950 shapes/distributions 下的 kernel/op correctness，不授权训练质量或 full-step 性能。

---

## 16. Resume conformance gates

### 16.1 exact_continue qualification

每个 certificate cell 至少使用 2 paired seeds、至少 2 个不同 quiescent checkpoint boundaries。每个 boundary 从同一 checkpoint 启动两个独立 resume branches，并与 uninterrupted branch 共同比较 100 个 post-resume optimizer steps；两条 resume 必须分别匹配 uninterrupted，也必须彼此匹配。

Uninterrupted 与 save/resume 分支逐 step 必须 bitwise 一致：

~~~text
sample IDs / data-order digest
routing fingerprint and immutable metadata digest
FP4 RNG tokens and domain frontier
packed-weight hash
loss / grad / BF16 model / FP32 master
optimizer / scheduler / scaler state
ledger leaf and hash-chain state
~~~

Load boundary 还必须 canonical bitwise 匹配 optimizer moments、consumed samples/tokens、router/expert-bias state 及 Python/NumPy/Torch CPU/CUDA RNG。

Packed cache 不进入 checkpoint；load 后 cache 必须 absent/invalid，首个 forward 从恢复后的 BF16 Parameters 重建，并与 fresh quantization bitwise 一致。

任一不一致使当前 exact run 立即失败；不得在该 run 内继续降级执行。只有另行通过第 16.2 节后，该 execution domain 才可声明 STATE_COMPLETE_ONLY；否则为 REPLAY_UNSUPPORTED。十步 replay 只能作为 smoke，不能用于正式 certificate。

### 16.2 state_complete_continue qualification

STATE_COMPLETE_ONLY 必须在与目标运行相同的 topology、rank mapping 和 serialized-state schema 上独立验证。每个注册 cell 至少 2 paired seeds、2 个不同 quiescent checkpoint boundaries，并满足：

- load boundary 对 canonical BF16 model、FP32 master、optimizer moments、scheduler、scaler、global step、consumed samples/tokens、data cursor、router/expert-bias state、committed weight epoch、ledger frontier 及所有已声明 RNG domain 做 bitwise 比较；
- checkpoint 中不存在 derived packed cache；load 后 cache 必须 absent/invalid，首次 forward 从恢复后的 BF16 Parameters 重建，packed bytes、scale bytes、layout IDs 与同一 snapshot 的 fresh quantization bitwise 一致；
- restore-state mismatch、缺失 state 或 cache rebuild mismatch 直接得到 REPLAY_UNSUPPORTED，不能标记 STATE_COMPLETE_ONLY；
- 每个 boundary 继续执行 100 个 successful optimizer steps；要求 finite loss/grad、ledger conservation、open=0、strict zero-fallback、无 OOM/device fault，并把 `continuation_stability_pass` 作为独立字段记录；
- 允许 post-resume 数值轨迹不 bitwise，但必须保存相对 uninterrupted branch 的逐 step loss/grad/model-state diagnostics；任何数值容差只属于 stability/quality protocol，不把 STATE_COMPLETE_ONLY 提升为 exact。

该门禁不授权质量声明。需要质量声明时仍按第 16.5 和第 17 节执行完整三臂矩阵。

### 16.3 reshard qualification

对每个显式 source-to-target topology tuple：

- load 前后 canonical gathered model、FP32 master、optimizer moments、scheduler、step 和 data cursor 必须匹配；
- derived cache 必须失效并从目标 rank 的 BF16 Parameters 重建；
- 2 paired seeds x 100 post-resume steps 通过 finite loss/grad、ledger conservation、open=0、strict zero-fallback、无 OOM/device fault；
- 该结果只认证该方向 tuple，不认证 bitwise replay 或反向转换。

### 16.4 Initial topology cells

实现计划至少注册并验证：

~~~text
exact:   4L  world8  TP8 PP1 EP8 ETP1 -> same
exact:   43L world16 TP4 PP4 EP4 ETP1 -> same
reshard: world16 TP4 PP4 EP4 ETP1 -> world16 TP8 PP2 EP8 ETP1
reshard: world16 TP8 PP2 EP8 ETP1 -> world16 TP4 PP4 EP4 ETP1
~~~

若资源 preflight 证明某 tuple 无法运行，必须在实现前以版本化 manifest amendment 替换为另一个明确 tuple；不能省略后仍声称 generic reshard support。

### 16.5 Quality inheritance

exact_continue 只有从 checkpoint 到已认证 quality endpoint 全程 bitwise identical 时，才可继承对应 uninterrupted cell 的质量结论。仅做 2x100 conformance 不自动获得质量声明。

state_complete_continue、warm_start 和 reshard_continue 若要获得质量声明，必须分别升级为第 17 节完整三臂矩阵。

---

## 17. Training quality gates

### 17.1 Experiment classes

| Model | Start mode | Precision arms | Minimum purpose |
|---|---|---|---|
| 4L | cold pretrain | BF16 / A8W4 / A4W4 | stability + powered short-horizon quality |
| 4L | warm_start | BF16 / A8W4 / A4W4 | stability + powered short-horizon quality |
| 4L | exact/reshard | each precision internally | resume conformance; quality only if upgraded |
| 43L | every supported mode | BF16 / A8W4 / A4W4 | 1 seed x 100-step stability |
| 43L | each publicly claimed cell | BF16 / A8W4 / A4W4 | powered short-horizon quality |

100-step run 只证明 stability、finite、routing、ledger 和 zero-fallback。它不是 convergence。

### 17.2 Pilot first, then preregistration

5 seeds × 500 updates 仅为 pilot 起点，不是正式 seed/step 数或最终质量门槛。首先运行 BF16 多 seed pilot 估计自然方差（initialization/routing/data 轨迹变化），再用独立 paired pilot 估计 A4-BF16、A8-BF16 与 A4-A8 差异方差及 evaluation noise。

在正式结果可见前预注册最终 seed list/n、step 与累计 valid-token horizon、NLL margins、per-seed caps、alpha/multiplicity、power target、exclusions 和 validation token 数。不得把旧版 n>=5、500 updates、0.010/0.005 NLL、0.030/0.015 caps 直接沿用为已批准硬门槛；它们仅是被 R19 supersede 的规划示例。Pilot 与 formal data 分离，pilot 不并入正式统计。

功效分析可用 `n ≈ ceil(((z_alpha + z_power) * sigma / margin)^2)` 作规划，再预注册与样本假设相符的 paired analysis。预算不足或无充分功效只报 smoke/short-horizon regression，不报 confirmatory non-inferiority、convergence 或最终质量。

### 17.3 Pairing contract

同一 seed 的三臂共享：

- pre-quant initial checkpoint；
- optimizer/scheduler/batch geometry；
- fixed cumulative non-padding token horizon；
- data order、tokenizer、preprocessing 和 validation corpus；
- dropout/router 等公共 RNG streams。

DataLoader 使用独立 rank-local generator。训练数据顺序、validation token/label/mask 和 preprocessing MUST 记录 digest。量化 RNG 使用独立命名 domain，不能扰动公共 RNG。

只有明确 infra-invalid attempt 可用同 seed 重跑；所有 attempts 均保留。NaN、发散、质量差、fallback 或 ledger violation 是实验失败，不能 clean rerun 到通过。

### 17.4 Primary metric and endpoint

Primary metric 是固定 held-out corpus 上、预注册数量的 valid target tokens 的 token-weighted validation NLL：

~~~text
NLL = sum(cross_entropy over valid targets) / number of valid targets
~~~

Pilot 估计 validation sampling noise；目标 half-width 与最终 token 数在 formal run 前冻结，旧 0.001 / 1M 仅作规划示例。

Endpoint 是固定累计 non-padding training-token horizon 对应的 checkpoint；最终 updates 数由 pilot 后预注册，并以实际 valid token 数对齐；500 updates 仅 pilot 起点。禁止选择 best checkpoint。更多 validation tokens 只降低 evaluation noise，不能替代独立 training seeds。

Primary evaluation 使用各 arm 当前 BF16 Parameter 的共同 deterministic BF16 path，以隔离学到的 checkpoint 质量。Native A4/A8 deterministic evaluation 是 required secondary。若未来声明 native inference quality，则 primary 与 native gate 均须通过。

### 17.5 Statistical gate

对 paired seed s：

~~~text
d_s = endpoint NLL(arm_1, s) - endpoint NLL(arm_0, s)
U95 = mean(d_s) + t_(0.95,n-1) * sd(d_s) / sqrt(n)
~~~

三个结果分开记录，使用 pilot 后正式预注册的 margin，而非旧固定数字：

~~~text
a4_vs_bf16_noninferior: U95(A4-BF16) <= margin_a4_bf16
a8_control_vs_bf16_valid: U95(A8-BF16) <= margin_a8_bf16
a4_vs_a8_activation_penalty_within_margin: U95(A4-A8) <= margin_a4_a8
quality_bundle_pass = AND(all three) plus preregistered safety caps
~~~

Per-seed safety caps 同样由 pilot 后预注册，不替代 power/CI。Conjunction 的 intersection-union 与独立宣传结论的 multiplicity 处理须预先选择，不在看到 formal data 后改变。

### 17.6 Claim boundaries

- 4L 不外推 43L；
- cold 不外推 warm/exact/reshard；
- 一个 topology 不外推另一个；
- pretraining 与 continued-training 不互相授权；
- exact_continue 只有 checkpoint 到 endpoint bitwise identical 才可继承 uninterrupted 质量结论；
- 其他 resume cell 若需要质量声明，必须升级为完整 BF16/A8/A4 powered matrix。

---

## 18. Benchmark and performance gates

### 18.1 General protocol

- AITER microbenchmark 调 public wrapper；Lumen benchmark 调真实 production API。
- 必须断言实际 selected backend/config/precision，strict results 的 fallback counters 为零。
- GPU timing 使用 CUDA events；distributed full-step 使用 max-rank wall clock。
- correctness、quality 和 performance artifacts 使用同一 clean/pinned manifest。
- tuning/timed path 禁止首次 JIT、lookup、autotune、扩容和 fallback probing。
- cold cache build、hot cache hit、quant、Fprop、DGrad、WGrad、full local expert block 和 full step 分开报告。

### 18.2 Workloads

从真实 BF16 与冻结后的 P1 strict DSV4 runs 捕获 routing artifacts，覆盖 4L EP8/local-E32 与 43L EP4/local-E64 的多 layer/step/seed。Benchmark 必须包含 bucket center/boundary、p10/p50/p90/p99 padded work、balanced、many-empty、single-hot 和 long-tail。

真实 route-weighted local-block latency 使用 final held-out trace 中每种 invocation 的出现频率加权；不得对已按 invocation 计数的结果再次按 token 数加权。

对 route artifact r，令 w_r 为归一化 invocation frequency，L(v,r) 为 variant v 的 fresh-process median latency。主 local-block speedup 唯一定义为：

~~~text
weighted_latency(v) = sum(w_r * L(v,r))
speedup(P1 over P0) = weighted_latency(P0) / weighted_latency(P1)
~~~

Tail 指标唯一定义为：按 w_r 展开/加权的 route-latency empirical distribution 上分别计算 Q0.9，再取：

~~~text
p90_latency_ratio = Q0.9(L(P1)) / Q0.9(L(P0))
~~~

Formal local-block interval 使用同一组 process-first paired bootstrap draws 同时重算 route weights、每个 route 的 paired latency statistic、weighted_latency/speedup 和 weighted-route p90 ratio。每次 draw 先重采样独立 capture group/process pair，再在 process 内按冻结的 contiguous block length 重采样；rank、route 或单次 kernel iteration不得被当作独立 provenance unit。LCB/UCB 分别取 paired bootstrap distribution 的 5th/95th percentile。

### 18.3 Microbenchmark statistics

每个正式 bucket/config 至少：

- 3 个独立 fresh process pairs，默认 5；最终 `n_micro` 由独立 pilot 的 paired log-throughput variance 与目标 power 决定；
- 每 process 20 warmups 和 >=200 timed samples per candidate；
- AB/BA balance，或交错使用 ABBA/BAAB/ABBA；
- 报告 raw samples、mean、median、p95、p99、CV；
- process-first hierarchical paired bootstrap，再在 process 内 contiguous block resample；
- block length 由 pilot autocorrelation 冻结，默认 5，并报告 5/10 sensitivity；
- 记录 GPU/ROCm/PyTorch/AITER、kernel repr/config、artifact digest、memory、roofline/arithmetic-intensity evidence。

Micro 与 full-step 分别估计独立单位上的 paired log-speedup 标准差，不共享一个 n_perf。对 `j in {micro,full}`：

~~~text
delta_log_j = expected_mean_log_speedup_j - log(promotion_threshold_j)
n_j = max(3, ceil(((z_0.95 + z_power) * sigma_log_j / delta_log_j)^2))
~~~

要求 `delta_log_j > 0`，one-sided alpha=0.05、power >=80%（推荐 90%）。`n_micro` 的独立单位是 fresh process pair；`n_full` 的独立单位是 fresh distributed paired replicate block。默认 5 只有在上述 power 计算通过时才足够。Pilot 的 variance、alternative mean、power、n 与 seed rule 必须在 formal data 前冻结；pilot samples 不进入 formal interval。

### 18.4 Full-step protocol

4L 与 43L 每个正式 comparison 使用 `n_full` 个独立 fresh distributed paired replicate blocks，至少 3、默认 5 仅在第 18.3 节 power 计算允许时成立。每个 block 强制使用三个独立 fresh distributed launches，顺序固定为 `C_before -> V_candidate -> C_after`。三者从 byte-identical initial training checkpoint 开始，包括 BF16 model、FP32 master、optimizer、scheduler/scaler、所有 RNG state 和 data cursor；使用同一 seed/data slice、相同且预先固定的 warmup step count/warmup data prefix、相同且预先固定的 timed optimizer-step count 与其他 matched run settings。Derived caches 不在 checkpoint 中，各 variant 按自身固定 contract 重建。每个 launch：

1. backend/JIT/cache 预热；
2. 至少 20 个不计时 optimizer steps；
3. 至少 100 个连续、未 skip 的 timed optimizer steps；
4. 计时窗口不含 checkpoint/eval/profile；
5. 起止各一次 device synchronize + world barrier，窗口内无额外同步；
6. 保留每 rank raw step samples 和 max-rank window elapsed；
7. 主指标为 non-padding global tokens/s；
8. CI 只使用下述预注册 hierarchical paired block bootstrap，不允许在看到数据后改用 t interval 或另一种 estimator。

任一 formal candidate/control 的任何 runtime fallback、OOM、NaN/Inf、kernel failure 或 skipped update 使整个 formal replicate block 失败。任何未声明的 backend/precision/config 变化同样使 block 失败。不得删除单个 step 或 clean rerun 到通过。完整 runs 必须使用相同 data、GBS/MBS、sequence length、recompute、optimizer/offload、logging、stream policy 和与各 variant contract 相符的固定 tuning policy。

Control replicate drift 默认必须 <=2%。Peak allocated/reserved、cold build、p95 step latency 和 physical HBM headroom 单列；formal run SHOULD 保留至少 10% HBM headroom。

对上述强制 `C_before -> V_candidate -> C_after` schedule，control drift 定义为：

~~~text
control_drift = abs(C_after-C_before) / ((C_after+C_before)/2)
~~~

C 为同一 primary metric 的 control window estimate。任一 paired block 超过 2% 即无效并保留为环境失败 artifact，不能删除后继续统计。

本 spec 不再允许“等价 schedule”或事后选择 estimator。对 throughput/rate 指标，唯一 control estimate 与 paired effect 为：

~~~text
C_hat_i = sqrt(C_before_i * C_after_i)
d_i = log(V_candidate_i) - log(C_hat_i)
speedup_i = exp(d_i)
~~~

Formal CI 使用固定 10,000 次、seed 预注册的 two-level paired moving-block bootstrap：先以 paired replicate block 为单位有放回重采样；再对每个被选 block 生成一组 contiguous circular step-block indices，并把同一组 indices 同时用于 C_before、V_candidate、C_after 及其全部 ranks。Rank 始终作为一个 distributed launch 的联合观测，绝不独立重采样。每次 draw 重新计算各 launch 的 `max_rank_window_elapsed`、non-padding tokens/s、`C_hat_i` 和 mean paired log-speedup。lower/upper 95% bound 分别取 bootstrap distribution 的 5th/95th percentile，最终在 log 域聚合后 exponentiate。Pilot 冻结 block length、bootstrap seed-generation rule 与 `n_full`；formal data 不得改变它们。

### 18.5 Baselines

~~~text
BF16 grouped
current DSV4 blockwise FP8 diagnostic with truthful pass-level fallback counters
P0 sequential A4W4
P1 grouped A4W4
~~~

每个 variant 使用自己的显式执行契约，不能用一条 grouped/strict 规则覆盖所有 baseline：

- **P1 grouped A4W4:** `backend=grouped`、`strict=true`、`require-certified-bucket`；backend fallback=0、precision fallback=0，所有 completed obligations 均为 grouped MXFP4。
- **P0 sequential A4W4:** `backend=sequential`、`strict=true`；backend fallback=0、precision fallback=0，所有 completed obligations 均为 sequential MXFP4。它不受 grouped tuning policy 过滤。
- **BF16 grouped:** 固定并记录 BF16 grouped backend/config；不得发生未声明的 backend substitution。MXFP4 strict/fallback 字段不适用，executed precision 必须始终 BF16。
- **current DSV4 blockwise FP8 diagnostic:** 固定审计时的 current execution policy，逐 FC1/FC2 和 Fprop/DGrad/WGrad 记录 selected backend、executed precision 与真实 fallback reasons。已知的 ragged-WGrad BF16 路径必须如实计数，不能伪装成 FP8 success；只要出现任何 fallback，该 comparison 仅为 deployment diagnostic，不进入 formal promotion evidence。
- **current DSV4 blockwise FP8 formal control:** 只有在全部参与 pass 能以显式 strict policy、backend fallback=0、precision fallback=0 执行相同 frozen workload 时才成立；其 backend/config/precision signature 必须预注册。否则该 formal control 为 unavailable，而不是放宽零 fallback 要求。
- **matched A8W4 diagnostic:** 若进入性能比较，必须使用与 P1 相同的 grouped execution structure、显式 config 和 strict zero-fallback；否则只能单列 diagnostic。

A8W4 quality control 不进入 A4 性能主表；可单列诊断。Micro/kernel speedup 不等于 full-step speedup，full-step speedup不等于 time-to-quality。

### 18.6 Promotion thresholds

Experimental promotion-qualified 前必须同时满足：

- real-route weighted local expert block：P1/P0 throughput lower 95% bound >=1.20x；
- 4L full-step：P1/P0 tokens/s lower 95% bound >=1.10x；
- 43L full-step grouped-value：P1/P0 lower 95% bound >=1.10x；
- 43L full-step：P1/BF16 lower 95% bound >=1.05x；
- 若对应的 current blockwise FP8 formal control 可用，4L 与 43L 分别要求 lower 95% bound >=1.00x；若只有含 fallback 的 deployed control，则该 ratio 降为 diagnostic，不作为 promotion gate 或正式性能结论；
- A4W4 相对 matched A8W4 若对外比较：lower 95% bound >=1.00x；
- held-out p90_latency_ratio 的 upper 95% bound <=1.05；
- memory 同时满足第 9.4 节 budget 与 headroom；
- 每个 formal ratio 的 P1 candidate 侧都来自 explicit `backend=grouped`、`strict=true`、`require-certified-bucket`、zero-fallback；control 侧也必须 zero-fallback，并严格遵循第 18.5 节对应 baseline contract。含 fallback 的 current-FP8 结果只能进入单列 diagnostic artifact。

这些阈值在获得实测证据前均为 unverified gates，不是性能承诺。未达门槛时功能 MAY 保留 experimental/reference-only，但不得默认启用或宣传 speedup。V1 不预承诺 1.6x。

---

## 19. Repository ownership and planned changes

### 19.1 AITER

预期交付：

- versioned public facade/compiled ABI 与 typed capability reasons；
- numeric selector、packing/layout、dual-layout quantization；
- grouped Fprop/DGrad/WGrad；
- matched-scale A8 non-weight operand support needed by the quality control；
- metadata builder/validator；
- caller-owned workspace contract；
- gfx950 tuning/bucketizer/config；
- public-wrapper unit tests、benchmarks、repr 和 artifacts。

所有新 GPU kernels 位于 AITER 正确目录。任何可复用 activation/shuffle/reduction/helper 放 utils 或 common，不复制近似实现。AITER 提交遵守 DCO；每个 kernel change 带 wrapper、test、benchmark 和必要 config。

### 19.2 Lumen

预期交付：

- DSV4 专用 CLI 与 conflict checks；
- P0 sequential 与 P1 grouped dispatch；
- RoutingMetadataV1 / BoundExecutionV1 / late attachment；
- autograd 与 direct FP32 main-grad WGrad；
- committed weight epoch、cache allocator/budgets；
- Philox domain allocator；
- obligation ledger 与 all-rank telemetry；
- warm/exact/state-complete/reshard checkpoint integration；
- module/op/model tests、training harness 和 full-step benchmark。

Lumen 不新增 GPU kernel，也不持久化 AITER tuning tables。

### 19.3 Megatron

V1 SHOULD 通过现有 Lumen provider/hook 接入，不修改 Megatron core。若 routing metadata 生命周期无法在现有 hook 表达，任何最小 Megatron patch 都必须先单独审查，且不能把 MoE 基础设施复制进 Lumen。

---

## 20. Milestones and exit criteria

### M0 — clean prerequisites

- [ ] Lumen/AITER dirty diff inventory 完整。
- [ ] 每个 prerequisite 是独立、可测试 commit。
- [ ] runtime AITER、gitlink 和 extension build-id 单一事实源。
- [ ] clean/pinned manifest，dirty=false。
- [ ] AITER clean baseline 建立在当前 upstream main 明确 commit（§23 R2 upstream-first 修订）；旧恢复成果保留，仅作逐项语义来源。
- [ ] mxfp4-moe implementation lineage 包含这些 commits；现有 docs commit 不被当作 code baseline。

### P0a — sequential operator reference

- [ ] --lumen-moe-mxfp4 --lumen-moe-mxfp4-backend=sequential --lumen-moe-mxfp4-strict。
- [ ] 与 P1 相同 quantizer/layout/RNG/output semantics。
- [ ] Fprop/DGrad/WGrad、empty/ragged、EP8/local-E32 和 EP4/local-E64 op gates。
- [ ] committed weight epoch/cache budget/checkpoint invalidation gates。
- [ ] 两个 FC shape 的 empty/tiny/nonaligned/skew 均做 dense MXFP4 F/D/W parity；日志分别记录 requested_backend=sequential、actual execution_mode=sequential、kernel_family、precision。

### P0b — fixed-random-init 4L training closure/regression smoke

- [ ] 固定随机初始化 4L，paired BF16 对照，完成 forward/backward/optimizer、router probability gradient、recompute/accumulation 与 closed ledger 的 training closure/regression smoke；显式 sequential MXFP4、zero precision fallback。
- [ ] seed/model/optimizer/data 顺序及 masking/positions/loss token 集合冻结；100-step 为既有 smoke 候选长度而非用户批准具体 run 值，不宣称 convergence 或最终质量。

### P1 — grouped correctness merge

- [ ] AITER public ABI、kernel tests、benchmark 和 config 完整。
- [ ] P1 grouped-vs-P0 parity 通过。
- [ ] bind transaction、late attachment、typed errors 和 no-post-commit-fallback 通过。
- [ ] direct FP32 main-grad、eager WGrad、full recompute 通过。
- [ ] 使用显式 validated config 完成 4L EP8 strict 100-step stability，逐 pass zero fallback。

### M2 — forward-ported release candidate

- [ ] ABI freeze 后 forward-port 到一个明确 pinned 的 latest-AITER commit，并更新 Lumen gitlink/adapter。
- [ ] 新 manifest 记录 source/tree、extension SHA256/build-id、ABI/layout/toolchain，且 dirty=false。
- [ ] 重新通过 numeric/ABI conformance、AITER Fprop/DGrad/WGrad correctness、P0/P1 parity 和 4L EP8 grouped strict 100-step gate。
- [ ] 在 M2 lineage 上完成 route descriptor/bucketizer held-out certification。
- [ ] 在 M2 lineage 上完成 4L EP8 与 43L EP4 strict stability，以及 warm/exact/state-complete/reshard 对应 conformance。
- [ ] M2 之前的 tuning、quality 和 performance artifacts 不进入 promotion evidence。

### Experimental promotion-qualified

- [ ] 所声明 quality cells 通过第 17 节 powered matrix。
- [ ] exact_continue cells 获得 run-level bitwise replay certificate。
- [ ] full-step/microbenchmark 通过第 18 节门槛。
- [ ] persistent/peak memory 在预算内。
- [ ] runtime compatibility、correctness、tuning 和 provenance 四类状态均为 true。
- [ ] 所有 artifact、raw samples、route data、seed/attempt ledger 和 manifests 可审计。

---

## 21. Failure semantics

以下类别必须结构化记录：

~~~text
candidate_miss
public_contract_error
runtime_operand_contract_violation
resource_error
kernel_error
asynchronous_device_error
distributed_abort
~~~

只有第 7.4 节列出的 candidate miss 可在 commit 前尝试下一个候选。OOM、workspace shortage、illegal alias、version mismatch、launch 后错误和 async fault 都不得 fallback。

建议细分 run-fatal reason：

~~~text
RUNTIME_SCHEMA_MISMATCH
ROUTING_EPOCH_MISMATCH
WEIGHT_EPOCH_MISMATCH
MAIN_GRAD_MISMATCH
WORKSPACE_SLICE_MISMATCH
SR_TOKEN_MISMATCH
ALLOCATOR_PLAN_BREACH
REPLAY_FRONTIER_MISMATCH
BOUND_KERNEL_FAILURE
~~~

任一 rank 的 strict failure 在下一个安全 collective boundary 传播 global abort。发生设备异步错误后不得继续提交 BF16 工作。

---

## 22. Risks and mitigations

| Risk | Consequence | Required mitigation |
|---|---|---|
| dirty prerequisites 未拆分 | 数值/性能变化不可归因 | M0 独立 commits + clean manifest |
| S1024 被误称 DSV4 S32 | 错误等价性声明 | 独立 recipe IDs/spec |
| row operand decode/requant | shape-dependent double quantization | same-BF16 snapshot golden tests |
| DCP 原位 load 不变 _version | stale packed weight | committed epoch + cache generation |
| late dY alignment 不满足 | commit 后被迫 fallback | bind-time staging reservation or candidate miss |
| route bucket collision | config ranking 翻转 | raw descriptor + held-out regret audit |
| hidden BF16 fallback | 假成功 | strict ledger and pass-level counters |
| SR 与 backend/attempt 绑定 | retry/recompute 不可复现 | canonical logical allocation + non-reclaimed ranges |
| deferred WGrad 扩大生命周期 | exactly-once/checkpoint 风险 | V1 fail at startup |
| 43L cache resident cost | OOM/吞吐不稳 | 0.75x default, 1.0x hard max, allocator quota |
| low-powered quality run | 伪 non-inferiority | BF16 variance pilot, preregistered n/horizon/margins, claim downgrade |
| path/commit 与 loaded binary 不同 | provenance 失真 | extension digest/build-id in manifest |

---

## 23. Decision register

本版本已整合全部需求确认：

~~~text
Q1  minimal clean prerequisites before implementation
Q2  gfx950, ETP1, EP8/local-E32 and EP4/local-E64, two DSV4 shapes
Q3  Lumen A4W4; explicitly not DSV4-report equivalent
Q4  expert-sorted tensor boundary
Q5  P0 -> P1 -> promotion with strict zero-fallback gates
Q6  AITER kernels/API; Lumen lifecycle/integration
Q7  --lumen-moe-mxfp4 naming
Q8  experimental pretraining/continued-training positioning
Q9  BF16 model Parameter quant source
Q10 dual layouts independently quantized from same BF16 snapshot
Q11 WGrad OVERWRITE / ACCUMULATE_FP32
Q12 one clean source/build lineage
Q13 Python facade vs versioned compiled ABI
Q14 query -> prepare -> bind -> launch
Q15 routing semantics/lifetime vs physical schema ownership
Q16 auto+strict is legal; selection separated from precision fallback
Q17 no V1 dynamic-routing graph replay guarantee
Q18 compatibility separated from certification
Q19 training/operand-quant/operand-transform recipe 与 physical/prepared layout IDs 分层
Q20 successful transactional bind is commit point
Q21 committed weight epoch is authoritative
Q22 0.75x persistent target, 1.0x hard ceiling
Q23 eager WGrad only
Q24 warm_start / exact_continue / reshard_continue
Q25 matched fixed-snapshot and full-training A8W4 controls
Q26 grouped-only tuning policy and multi-axis certification state
Q27 obligation conservation ledger
Q28 late runtime operand attachment only
Q29 planner + allocator memory enforcement
Q30 deterministic rank/domain Philox allocation
Q31 run-level replay certificate and hash-chain frontier
Q32 matched-scale A8W4 format lift
Q33 stable seghist descriptor; data-derived bucketizer
Q34 power-determined paired quality matrix and scoped claims
~~~

2026-10-09 用户批准修订（执行顺序与本轮范围；不改变上述数值/ABI/lifecycle contract）：

~~~text
R1  本轮执行范围 M0 + P0：M0 clean-lineage 与 P0 sequential A4W4 executable
    numerical reference；P1 及以后阶段（grouped、T0/C0、QG、B1、promotion）
    不在本轮实施，其完整契约（§5–§19）不删减、不提前开发
R2  upstream-first：AITER clean baseline 直接建于当前 upstream main 明确 commit
   （本轮核查锚点 aa2bdc6273adb900ffcbdedfc192d5d6a65bf261，不假定持续最新），
    拟建独立 worktree /scratch/users/xdai/aiter-mxfp4-moe（本记录时不创建），
    面向最终规范上游 PR；旧恢复成果（e35bb17 基线 + 21 条恢复文件）保留，
    仅作逐项语义来源，不作整包 PR 基线；取代旧"先在旧 M1 完成、后 M2
    forward-port"顺序。修订原因：直接在可上游 PR 的 clean lineage 上实现，
    避免在不可审查的混合 dirty 基线上实现后整体前移
R3  HC_MULT=4 为本轮训练 fixture 确认值（与官方 DeepSeek-V4-Flash config
    hc_mult 一致）；旧脚本默认 2/4 的差异以此为准
R4  固定 digest 基础镜像允许适配
   （docker.io/zhangdanyangamd/lumen@sha256:d398f63465ac9cb5219f52f9a42993e109b9efc37115047d4d08536eddb4b436）；
    允许独立 pinned Megatron 副本处理版本冲突，不改镜像原件；源码独立挂载
    保证唯一导入/编译来源；实际 pin/patch 经环境核实后 freeze
R5  C4 采样（allenai/c4 en 前 614475 train / 前 21792 validation、train 原序
    repeat4）与无历史文件事实已确认；tokenizer 拟采用官方
    deepseek-ai/DeepSeek-V4-Flash pinned revision（本轮核查
    60d8d70770c6776ff598c94bb586a859a38244f1，实施前重新核实），
    vocab 129280 与既有配置一致；实际加载验证未完成，不作为已验证事实
R6  验收按获批资源分档；每次资源申请单独批准
~~~

本轮 R19 已关闭随机初始化选择、packing 隔离、router trainability 与 FSDP disposition；具体 4L 超参/LR/optimizer/GBS/MBS/seq/checkpoint 频率仍待运行 manifest 冻结，镜像拉取、tokenizer/C4 实际下载、资源申请与 commit/push/PR 均未授权。本次同步仅为文档记录，不构成实施、下载、资源或提交批准。

2026-10-09 GPT 评审修订（R7–R18，已裁定接受；仅文档修订，不改变未冻结项的待决状态）：

~~~text
R7  头部状态改分层：M0 frozen；P0a operator contract frozen；P0b training
    fixture decisions now closed by R19; run manifest pending；P1 architecture reviewed；P1 compiled
    ABI draft；validation protocol correctness frozen、quality/performance
    pending preregistration；execution authorization not granted；
    implementation not started
R8  P0 拆分为 P0a（operator reference，契约已冻结）与 P0b（training
    closure/regression smoke，选择由 R19 关闭）；修正旧"没有问题阻塞 M0/P0"
    表述；新增 Open Decisions 表
R9  no-CK 契约：本 feature 的 quantization/cache preparation/Fprop/DGrad/
    WGrad MUST NOT 选择 CK（Composable Kernel），包括内部 auto-dispatch；
    kernel_family ∈ {FLYDSL, TRITON, ASM}（Gluon 不在 allowlist）；
    execution_mode {grouped,sequential} 与 kernel backend 命名分离；
    capability/BoundExecution/manifest/ledger/kernel repr 记录
    kernel_family；strict gate 每 rank 断言 ck_invocations=0；P0 现有
    Dense MXFP4 调用链同样审计
R10 新增 Lumen framework adapter contract（第 11.4 节）：GroupedMLP 兼容
    forward 签名、per-expert Parameter/state_dict/recipe/cache、custom
    autograd、WGrad exactly-once 直接累加 FP32 main_grad、main_grad 分配
    时机、Megatron bookkeeping 归属、accumulation window 归属、reducer
    完成信号、Parameter 顺序与 local expert id 映射
R11 历史建议已被本轮 R19 supersede：FSDP V1 out of scope
R12 P1 compiled ABI 标注 draft；新增精确 ABI 字段要求（第 7.6 节）
R13 recipe ID 分层修正：numeric ID 改 scale_block_32x32 /
    scale_block_1x32，避免与物理 tile 混淆；H16 从 physical layout 改列
    为 operand transform（数值变换）；引入 training_recipe_id /
    operand_quant_recipe_id / operand_transform_id 分层；row1x32 经
    dense 源码核对为非 weight operand 的 1x32 scale recipe，保留为
    operand recipe 并澄清角色；全部交叉引用同步更新
R14 H16/WGrad 数学修正：逻辑公式改 Q(H16·pad(dY_e))^T [N,P_e] ×
    Q(H16·pad(X_e)) [P_e,K] → dW_e [N,K]；明确 X/dY 同一 H16、
    normalized 约定与 1/16 compensation 条件、先补逻辑零再 H16、
    pre-transform padding lanes=+0 而 post-transform lanes 参与
    量化/GEMM、all-plus 指 diagonal sign vector 全正
R15 strict 与 certification 分轴：runtime_compatible / strict /
    tuning_policy 三轴独立；promotion/exact_continue 额外要求
    certified；uncertified 不等于 runtime incompatible，
    allow-validated-heuristic 可用
R16 动态 M_e 运行时合同（第 6.4 节）：唯一权威来源为 EP dispatch/local
    sort 后 local tokens_per_expert；counts/offsets dtype/device；
    expert id ↔ weight slot 映射；可选 active_expert_ids 压缩；P1 hot
    path 禁止逐 expert .item() 与新增 D2H 同步；max_M_e/ΣM_e/offset
    宽度/capacity 上限与超限 typed failure；extreme token count 公式
R17 cache 预算拆分 persistent_cache_bytes / peak_cache_build_bytes /
    reusable_workspace_bytes / transient_launch_bytes /
    device_total_preflight_bytes 分别报告限制；新增 test-only
    cache-disabled P0 oracle 与 cache-enabled parity
R18 配套 plan 同步修订：分层状态、P0a/P0b、Open Decisions、no-CK 规则、
    ABI draft 标注
~~~

明确不做：文件拆分、全量 Requirement ID 追溯（后续单独处理）；质量数字（如 5 seeds / 500 updates）保持候选；FSDP V1 out of scope（本轮 R19 supersedes R11）。

### Closed decisions and remaining run-manifest fields

| ID | 本轮已拍板（closed） | 仍需记录但不得擅自填成已批准 |
|---|---|---|
| OQ-P0-01 | 接受固定随机初始化 4L，仅 training closure/regression smoke；固定 seed/model/optimizer/data 顺序，paired BF16 对照 | seed 具体值、optimizer/LR/schedule/GBS/MBS/seq、checkpoint/recompute 具体配置和 smoke 长度进入运行 manifest；不另设随机/预训练初始化选择题 |
| OQ-P0-02 | 强制跨文档 attention 隔离；BF16/MXFP4 完全相同 data、attention mask、positions 与 loss token 集合 | EOS 插入/position reset 的具体实现固定并测试；通过文档边界和显式有效长度/target mask 区分 padding 与真实 EOS，EOS=pad 时禁止仅按 token ID 屏蔽真实 EOS；两臂 valid-token digest 必须一致 |
| OQ-P0-03 | router 保持原精度/BF16、可训练，保留 routing probability gradient；仅 routed FC1/FC2 GEMM 量化 | gate Parameter 梯度与 e-score bias 的非梯度 load-balancing 状态更新分别记录、测试和 checkpoint；不得把 bias 状态误称可训练梯度或冻结整个 router |
| OQ-G-01 | V1 明确排除 FSDP；reshard 仅 Megatron/DCP；未来接口引用保留但不进 exit criteria | 任何未来 FSDP 支持另案设计/验收 |

设计选择已关闭不等于允许运行：具体 manifest、数据准备与环境核实和实施/训练/下载/资源/提交授权仍未完成。不得重问上述已决选择，也不得替用户发明具体值。

---


### R20 — Execution selector / dense reuse contract clarification

本轮保留唯一待实现 CLI 名称与原建议默认；新增 §3.6.1 dense pass-by-pass reuse prerequisite、§7.7 sequential/grouped 分层职责与同 recipe/snapshot 语义、§8.2 fallback 计数真值、§15.5 测试要求；Plan §7–§10 同步。P0a/P0b 不依赖先交付 grouped backward，仍要求 Lumen 完整 autograd/main_grad/router-gradient/lifecycle 闭环；P1 继续为后续 grouped 扩展。R19 源码事实、SonicMoE 反证及 QAT 非兼容目标/非官方复现/experimental 上限全部保留；本轮不授权代码、提交、下载、GPU 或资源操作。

### R19 — 本轮一手源码 reconciliation（supersedes 与章节映射）

本轮只改 Spec/Plan 并同步副本；不启动开发，不提交/推送、不改分支/worktree、不下载模型/C4/tokenizer、不安装拉镜像、不编译/运行 GPU、不申请/变更资源。

| 变更类别 | Supersedes | Spec 落点 | Plan 落点 |
|---|---|---|---|
| 事实修正 | 泛称 current FP8 为 grouped、以 issue 证明所有低精度 backward 缺失、把 forward fusion/PDL 当训练能力 | §1、§3.5–3.8 fixed-source map/capability matrix；§4.3/§5 数学保持冻结 | §7 A0 reuse audit、§9 A4 K-grouped reference、§10 wrapper boundary |
| 阶段变化 | R7/R8 中 P0b 初始化/packing/router 未决的历史描述；R11 FSDP 建议未决 | §1、§2.2、§20、§23 closed decisions | Status、§1、§8 P0a/P0b、§10、§22 closed decisions |
| 验收标准变化 | 旧 5×500 与固定 NLL/cap 数字作为最终门槛；混用 sequential 主动路径与 fallback | §8/§12 telemetry、§17 BF16-first pilot/preregistration、§20 | §8、§15 QG、§16 B1；每 rank/pass zero-fallback 必要但不充分 |

固定 main evidence pins 只反映本次阅读，并不证明远端 import/build 已对齐或测试通过。旧 R1–R18 保留历史来源；与本轮 closed decisions、pilot-only 数字和 source matrix 冲突的表述以 R19 及已同步正文为准。

---

## 24. Key source index

| Topic | Source snapshot location |
|---|---|
| DSV4 model/expert args | examples/dsv4/dsv4_megatron_args.sh |
| DSV4 grouped provider | lumen/models/dsv4/megatron/spec_provider.py |
| Megatron grouped expert flow | megatron/core/transformer/moe/experts.py |
| Token dispatch/sort/combine | megatron/core/transformer/moe/token_dispatcher.py |
| Current DSV4 FP8 enable | lumen/models/dsv4/megatron/fp8.py |
| Current sequential grouped module | lumen/modules/grouped_linear.py |
| Existing grouped GEMM dispatch | lumen/ops/gemm/grouped_gemm.py |
| Dense MXFP4 Fprop/backward | lumen/ops/quantize/linear.py |
| Dense MXFP4 quantization/SR | lumen/ops/quantize/ops.py |
| Dense MXFP4 cache/invalidation | lumen/quantize/__init__.py |
| Lumen backend dispatch | lumen/ops/dispatch.py |
| AITER DSV4 A8W4 tuned rows | aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv |
| AITER current MoE WGrad | aiter/ops/triton/moe/moe_wgrad.py |
| Lumen benchmark utilities | benchmarks/bench_utils.py |
