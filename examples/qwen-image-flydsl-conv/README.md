# Qwen-Image / Wan2.1 接入 Lumen FlyDSL 卷积 —— 集成验证报告与复现手册

> English version: [README_EN.md](README_EN.md)

本文记录图像与视频两条算子集成验证：把冻结的 Qwen-Image / Wan2.1 VAE 中的卷积改由 Lumen 的
FlyDSL implicit-GEMM 内核承担，在真实的 8 卡 VeOmni 训练中运行，并与未接入的
同配置运行逐项对照。第一至六节是 Qwen-Image 的报告与复现步骤，
[第七节](#wan-results) 是 Wan 视频路径、训练与离线 embedding 的实测结果。

**测试覆盖（2026-09-09 核对）**：Qwen-Image 已完成 256×256 下 2 / 10 / 500 步
全参数 SFT、FlyDSL 10 步重复对照，以及 1024×1024 下独立复现与 10 步 A/B；
500 步 mean loss 从 0.02083 降至 0.00621（−70.2 %），但数据仅为 8 张循环图像，
这是优化链路与过拟合验证。Wan 已完成 17 / 81 帧单卡验证，以及 18 次 10 步正式对照
和 6 次 VAE 计时诊断；尚无 Wan 长程收敛或生成质量评估。

---

## 一、结论

在 8× AMD Instinct MI355X（gfx950）、1024×1024 训练配置下，三项目标均已达成，
另有一项必须同时说明的边界。

| # | 结论 | 判据 | 结果 |
|---|---|---|---|
| 1 | **FlyDSL 3D 卷积确认启用** | 训练日志出现 `backend for conv2d: FLYDSL`，且 52 个 `QwenImageCausalConv3d` 被接管 | ✅ 达成 |
| 2 | **计算结果正确** | 以 FP32 编码为基准，接入后 **51.7 dB**，原生 BF16 为 51.6 dB | ✅ 精度未劣化 |
| 3 | **端到端性能提升** | 完整一次 VAE encode：**33.05 ms → 17.22 ms，加速 1.92 倍** | ✅ 达成 |
| — | **训练步时无可测变化** | VAE 卷积仅占单步 0.56 %，而同配置重复运行波动达 ±6 % | ⚠️ 见第四节 |

### 1. 算子确认启用

Lumen 的调度层在算子能力不匹配时会**静默回退到 torch**，此时训练照常结束、
loss 也正常，与"集成成功但恰好无变化"在外部表现上完全一致。因此本次集成不以
运行成功作为判据，而是在训练入口显式输出后端归属：

```
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

第二行是内核确实执行的唯一直接证据。这条日志在本次工作中确实拦下过一次真实的
静默回退（详见第五节 BF16 一项）。

### 2. 结果正确性

以 FP32 编码作为参照基准，而非要求逐位相同——更换卷积算法必然改变归约顺序，
BF16 下的舍入差异是预期内的。

| 对照项 | 信噪比 |
|---|---|
| 原生 BF16 模型 vs FP32 | 51.6 dB |
| **接入后 BF16 vs FP32** | **51.7 dB** |
| 单层：conv3d 重写为 conv2d（仅算法变换） | 90.4 dB |
| 单层：torch conv2d vs Lumen conv2d（仅内核差异） | 51.0 dB |

第三行说明因果重写本身是干净的，全部差异来自内核的归约顺序，属 BF16 舍入量级。
训练侧逐步 loss 偏差均值 0.945 %，而同一配置重复运行的偏差本底即为 0.28 %，
两者同量级。

### 3. 端到端性能

| 范围 | 未接入 | 接入后 | 加速比 |
|---|---|---|---|
| **完整 VAE encode** | 33.05 ms | 17.22 ms | **1.92×** |
| 仅卷积部分，NCHW | 18.76 ms | 6.12 ms | 3.07× |
| 仅卷积部分，NHWC | 18.76 ms | 3.78 ms | 4.97× |

其中 encode 一项由 `verify_vae_patch.py` 直接计时、内部多次重复取平均，
是本文可信度最高的性能数据。

需要说明收益的实际来源：主要并非"FlyDSL 的卷积比 torch 快"，而是 torch 的
`conv3d` 在时间维退化为 1 的张量上效率很低，降维成 2D 卷积贡献了大部分收益，
FlyDSL 内核补足其余。FP32 下仅做重写、完全不涉及 FlyDSL，一次 encode 即可从
9.474 ms 降至 4.180 ms。

### 4. 结论的边界

**本文不主张训练加速。** VAE 卷积在单步中占比 0.56 %，理论最优收益约
0.44 %，而同配置重复运行的步时波动为 ±6 %，噪声高出一个数量级以上。
详细论证与实测反例见第四节。

本次工作定位为**正确性与链路打通的里程碑**，是后续算子与 Lumen 优化的度量基线。

---

## 二、环境要求

- 硬件：单节点 8× AMD Instinct MI355X（gfx950），ROCm 宿主驱动
- 磁盘：约 150 GB（镜像 76 GB + 模型 54 GB）
- 耗时：约 40 分钟，主要为模型下载与两次约 4 分钟的训练

默认训练分辨率为 **1024×1024**，与官方 `configs/dit/qwen_image_sft.yaml` 一致。
提高分辨率在本机的额外开销可以忽略（步时 3.95 s → 4.01 s，显存增加 1 GB），
理由见第四节。如需 256×256 的小配置，运行时设 `RES=256` 即可。

---

## 三、复现步骤

### 0. 准备工作目录与容器

所有内容集中在宿主机的一个目录下，bind mount 到容器的 `/work`，需 150 GB 空闲空间。

```bash
export ROOT=/mnt/nvme/$USER/flydsl-demo      # 按实际情况修改
mkdir -p "$ROOT"

docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12
# 期望 digest：
# sha256:fb0c497921af3df3faae09c96712d375b0b690dc05349f0c8e638ed377c8c956

docker run -d --name lumen-flydsl-demo \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc host --shm-size 32g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1 \
  -e FLA_TILELANG=0 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$ROOT":/work \
  -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity
```

后续命令统一使用以下简写：

```bash
dx() { docker exec -w /work lumen-flydsl-demo bash -lc "$*"; }
dx 'rocm-smi --showid | head -12'      # 应看到 8 张卡
```

⚠️ `HIP_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 必须同时设置，
只设其一会直接中止并报 `Conflicting visibility of agent-N`。

### 1. 获取代码

```bash
dx 'git clone -b dev/flydsl-conv3d https://github.com/ZhangDanyang-AMD/Lumen.git /work/Lumen'

# VeOmni 固定在本例验证过的 commit。后续 commit 可能移动
# train_dit_lumen.py 所包装的 trainer hook 位置。
dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    cd /work/VeOmni &&
    git checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f &&
    git status --short && echo "(无输出即为干净)"'
```

此后路径统一由 `env.sh` 提供：

```bash
export EX=/work/Lumen/examples/qwen-image-flydsl-conv
dx ". $EX/env.sh && echo \$LUMEN_PYTHONPATH"
```

**自此两个仓库均不再有任何文件改动。** 每次训练都会记录双方的
`git status --short | wc -l`，应始终为 `0`。

### 2. 配置环境

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

包含三项操作，均幂等：

1. 将 flydsl 0.3.2 安装至独立目录（`/work/pyenv/flydsl-0.3.2`）
2. 补齐 aiter 缺失的 9 个文件（Lumen 导入所需）
3. 以 `--no-deps` 方式 editable 安装 VeOmni

前两项的必要性见第五节，两者均不修改镜像自带的任何文件。

期望输出结尾：

```
torch       2.12.0+rocm7.14.0a20260608
torch.hip   7.14.60850
torch.cuda  None (must be None on ROCm)
devices     8
flydsl conv3d probe: True (must be True)
SETUP OK
```

> ❌ 禁止执行 `uv sync`、`pip install -e '.[gpu]'`，禁止安装或升级
> torch、torchvision、triton 及任何 `cu12`/`cu13`/`nvidia-*` 包。
> 任一操作都会把 ROCm torch 替换为 CUDA 构建，导致环境不可用。
> ❌ 禁止设置 `HSA_OVERRIDE_GFX_VERSION`，gfx950 被原生识别。

### 3. 模型与数据

54 GB 的模型下载是耗时主项，数据集在本地数秒内生成。

```bash
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && du -sh \$QWEN_IMAGE_DIR && ls \$QWEN_IMAGE_DIR"
```

应为约 54 GB，且包含 `transformer`、`text_encoder`、`vae`、`tokenizer`、
`scheduler`、`model_index.json`。

```bash
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"
```

80 = 10 步 × 8 卡，**记录条数是硬约束**。DiT 路径强制 `dyn_bsz=False`，
每个 rank 仅获得 `floor(N / dp_size)` 个 batch，数据耗尽时 trainer 直接跳出 epoch，
既不报错、退出码也为 0，只是静默少跑若干步。`run_10steps.sh` 已内置生成与校验，
本步骤可跳过，此处列出是为使该约束可见。

### 4. 单卡验证算子

三项单卡检查，合计约一分钟。按顺序执行，出现失败即停止——每项定位的问题不同。

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_lumen_conv.py"
```

验证 Lumen conv 算子自身：接线、对 FP32 的正确性、吞吐、autograd 路由、参数校验。
结尾应为 `PASS`：

```
frequency-weighted, one VAE forward (1024x1024)
  torch bf16 (baseline)    33.84 ms   1.00x
  lumen conv2d NCHW        18.74 ms   1.86x
  lumen conv2d NHWC        14.42 ms   2.44x
correctness: worst SNR 85.0 dB (threshold 25)
```

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/trace_vae_convs.py"
```

对真实的 `AutoencoderKLQwenImage` 挂 forward hook，列出一次 encode 实际调用的卷积
（16 种配置共 30 次）及逐层耗时。默认 1024，期望约
`torch 18.763 ms → lumen NCHW 6.115 ms (3.07x) / NHWC 3.778 ms (4.97x)`。

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype bf16"
```

验证 patch 本身，**这是最关键的一项**，结尾应为 `PASS`：

```
52 convolutions on FlyDSL
accuracy: 51.7 dB vs FP32, against 51.6 dB for the stock BF16 model
speed   : whole encode 33.050 -> 17.217 ms (1.92x)
conv2d locked to: FLYDSL
```

如需主动观察静默回退这一失败模式，用 FP32 运行同一脚本——这正是 VeOmni 加载
VAE 的方式：

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype fp32"
```

此时 backend cache 为空，且 `torch conv2d vs lumen conv2d : inf dB`——两者完全相同，
因为 FP32 下 Lumen 就是 torch。原因见第五节。

### 5. 两次训练，各 10 步

每种模式各运行一次，约 4 分钟。默认 1024。

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
```

两次运行共用同一入口、配置、数据、随机种子、并行度与解释器环境，
**仅 `LUMEN_PATCH` 不同**，因此任何差异都可归因于 patch 本身。

前约 110 秒为模型加载、FSDP2 wrap 与 RCCL 初始化，step 1 另需 70–90 秒进行
kernel 选型，**不要提前终止**。稳态约 4 s/step，单次运行总计约 250 秒。

flydsl 模式必须输出：

```
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

**缺少第二行即表示内核未执行**，转第六节故障排查。

如需将 BF16 转换与内核收益分开计量，可另跑一次 `bf16-only` 模式。
如需 256×256 对照，设 `RES=256`（产物名带分辨率，不会相互覆盖）。

### 6. 对比

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/compare_runs.py baseline-1024 flydsl-1024"
```

在 8× MI355X 上完整执行上述步骤的实际输出（数值会有出入，关注量级而非末位）：

```
=== did the patch actually run? ===
  baseline-1024  no [lumen] patch lines (this is expected for the baseline)
  flydsl-1024    vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
  flydsl-1024    backend for conv2d: FLYDSL

=== total_loss per step ===
 step       baseline-1024         flydsl-1024
    1         0.005066125         0.005102025
    2         0.007112541         0.007285969
   ...
   10         0.005530797         0.005613535

=== loss deviation from the control ===
  flydsl-1024  max 2.440%   mean 0.945%

=== timing ===
  mode            step 1    steady     wall
  baseline-1024     73.8s     4.01s     246s
  flydsl-1024       83.4s     3.91s     247s

=== peak memory (torch max_memory_allocated) ===
  baseline-1024   38.76 GB
  flydsl-1024     38.63 GB
```

**结果解读**

- **backend 行存在** —— 集成成功，这是单次对比唯一能确定的结论。
- **loss 偏差 0.945 %** —— 等同于未变化。同一模式重复运行的偏差本底为
  均值 0.28 %、最大 0.86 %，本次结果与之同量级，两条曲线逐步贴合。
- **稳态步时** —— 单次对比不足以支撑结论，见第四节。
- **step 1** —— 含 kernel 选型，flydsl 另含 JIT 编译。某组测量中该开销约 12 秒，
  本组则完全不可见，两个数值均不宜引用。
- **峰值显存** —— 持平。BF16 VAE 省下的数百 MB 在 288 GB/卡的容量下不可见。

---

## 四、结果分析

### 单次运行的结论效力

对本文的前两个目标而言，每种模式运行一次即已足够——"内核是否执行"与
"数值是否改变"都是单次可判定的问题。

但对步时不成立，且差距悬殊：

| 项 | 数值 |
|---|---|
| VAE 全部卷积，一次 encode | 18.76 ms |
| 理论最优每步节省 | 约 15.0 ms，即单步的 **0.44 %** |
| 同配置重复运行的步时波动 | **±6 %** |

噪声高出效应一个数量级以上。本次工作中该陷阱两次成立：256 配置下曾有一组 A/B
显示接入后慢 8.8 %，形似性能回退，每种模式补做三次重复后两组均值几乎相同
（4.06 s vs 4.04 s）、区间完全重叠，实为漂移；1024 配置下 4 步的一组显示慢 17 %，
10 步的一组显示快 2.5 %，方向相反。迄今全部运行中，两种模式均落在
3.4–4.3 s/step，**无分离**。

因此：**不应从本文引用训练加速比。** 能够分辨内核收益的测量是
`verify_vae_patch.py`。若确需步时数据，每种模式至少运行三次并比较分布。

### 为什么采用 1024

两点原因：内核收益在 1024 下显著更大，而提高分辨率的代价在本机可以忽略。

同一份代码、同一台 8× MI355X，唯一差异为 `condition_model_cfg`
（`height`/`width` 256→1024，`max_sequence_length` 64→512）：

| 指标 | 256 | 1024 |
|---|---|---|
| 完整 VAE encode，接入前 → 后 | 4.44 → 3.83 ms（1.16×） | **33.05 → 17.22 ms（1.92×）** |
| 仅卷积，NCHW / NHWC | 2.23× / 2.67× | **3.07× / 4.97×** |
| 卷积占单步比重 | 0.068 % | **0.56 %** |
| 稳态步时（基线） | 3.95 s | 4.01 s |
| 峰值显存 | 37.79 GB | 38.76 GB |

256 下多数卷积层受 kernel launch 开销支配（普遍停在 30–40 µs 的下限），
内核的效率优势无从体现；1024 下张量规模足够，优势才显现出来。

**提高分辨率的代价低于预期。** 图像 token 由 256 增至 4096（16 倍）、
文本 token 由 64 增至 512，稳态步时仅由 3.95 s 变为 4.01 s，显存增加 1 GB。
原因是该训练步**受开销支配而非受算力支配**：MFU 约 0.2 %，GEMM 仅占步时约 1 %，
因此 16 倍的 token 计算量几乎不体现在总耗时上。

由此可以得出一个对后续工作更重要的判断：**在当前训练配置下，优化算子无法提升
训练速度**，99 % 的时间并不在算子中。若目标是缩短步时，应先对这 99 % 做 profile。

---

## 五、实现说明

### 两处必要的环境处理

**其一，独立目录安装 flydsl 0.3.2。** 镜像自带 flydsl 0.1.6，其 JIT 无法完成链接
（`could not find path component of main program: 'ld.lld'`）；0.3.2 可用，
但就地覆盖会破坏 aiter——aiter 依赖 0.3.2 已移除的 `fly_values`，
连带禁用其 CK 与 HIP ops。因此 0.3.2 安装至独立目录，仅在本例的 `PYTHONPATH`
中生效，镜像环境保持逐字节不变。VeOmni 不受影响，其源码对 `aiter` 的引用为零处。

一个可见的副作用：启用该目录后，diffusers 探测 attention 后端时
`from aiter import flash_attn_func` 会失败并打印 `Falling back to native attention`。
**基线运行因此必须使用相同的 `PYTHONPATH`**，将该差异排除在对比之外。
⚠️ 不要将 flydsl 运行与未启用该目录的运行相比较。

**其二，补齐 aiter 的 9 个缺失文件。** Lumen 在包级别导入了仅存在于其自有
aiter fork 的若干 triton 模块，上游 aiter wheel 不含这些文件，缺少时
`import lumen` 直接失败。`overlay_aiter.py` 只写入当前不存在的文件，
不覆盖 aiter 已有的任何内容，可用 `--revert` 完整回滚。该步骤与 conv 无关。

### 为什么 flydsl 模式同时转 BF16

单独使用 `LUMEN_PATCH=vae_conv` 不会触发任何 FlyDSL 代码：

> VeOmni 以 `torch_dtype=torch.float32` 加载 VAE
> （`modeling_qwen_image_condition.py:83`，硬编码），而 FlyDSL 内核仅支持 BF16。
> Lumen 的能力检查在进入 dispatcher 之前即判定不适用，全部调用转由 torch 承担，
> 不报错、不告警、backend cache 为空。

`vae_bf16` 将冻结的 VAE 转为 BF16，使内核可达。它**不属于 Lumen，也并非零代价**——
它改变了 trainer 的计算内容，因此设为独立开关，并保留 `bf16-only` 模式单独计量。
实测该项使 loss 偏离对照 1.28 %，约为复现噪声本底的 4.5 倍，而 conv patch 本身
停留在本底附近。

该转换不能直接进行。单独执行 `vae.to(bfloat16)` 会导致：

```
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

原因是 condition model 的 diffusion timestep 的 dtype 直接继承自 latents，
而 `scale_noise` 需在 FP32 时间表中按精确相等查找下标，BF16 无法匹配。
`train_dit_lumen.py` 因此包装了 `encode()` 使其返回 FP32 latents，
下游全部 dtype 与基线一致，唯一变量是 VAE 内部的卷积精度。

**使内核支持 FP32 是消除这一整套复杂度的唯一根本改动**，已列入后续计划。

### patch 的工作原理

`lumen_vae_conv.py` 重新绑定 52 个 `QwenImageCausalConv3d` 的 `forward`，
不涉及参数与模块结构，这是它在 `init_device=meta` 与 FSDP2 下安全的前提。

`QwenImageCausalConv3d` 在时间维做非对称 padding——前补 `kT-1` 个零帧、后不补——
再以 `padding=0` 卷积。当输入仅含一个真实帧时，除最后一个 kernel slice 外
全部与零相乘，因此

```
conv3d(pad(x), w)  ==  conv2d(x, w[:, :, -1])
```

为**精确恒等式**而非近似。对 3D 卷积使用对称 `padding=1` 并非 causal，结果不同。
凡不满足前提者（真实视频 `T > 1`、推理 feature cache、时间维含 stride 或 dilation）
一律回落到模块自身的 forward。

9 个模块**有意保留在 torch 上**：1×1×1 pointwise 卷积与 `time_conv` 层，
在这些层上 Lumen 反而更慢（0.85–0.94×），因其受 launch 开销支配。
模型原生的 `nn.Conv2d` 下采样层同理，未做替换。

---

## 六、附录

### 文件清单

| 文件 | 作用 |
|---|---|
| `env.sh` | 全部路径的唯一来源 |
| `setup_env.sh` | flydsl 独立安装、aiter 补齐、VeOmni 安装 |
| `lumen_vae_conv.py` | **patch 本体**，T=1 因果重写、T>1/cache 卷积替换与层选择策略 |
| `train_dit_lumen.py` | VeOmni 入口，`LUMEN_PATCH` = `vae_conv` / `vae_conv_video` / `vae_bf16` / `linear` |
| `run_10steps.sh` | 单次训练，`baseline` / `flydsl` / `bf16-only`，`RES` 选分辨率 |
| `run_wan_10steps.sh`、`wan_video.yaml` | Wan 全参数 SFT；`WAN_TASK=offline_embedding` 切换离线 embedding，同样支持三模式 |
| `compare_runs.py` | 多运行的逐步 loss、步时、显存对比 |
| `verify_lumen_conv.py` | 算子自身验证 |
| `verify_vae_patch.py` | patch 数值验证，及 FlyDSL 已执行的证据 |
| `trace_vae_convs.py` | 真实逐层卷积清单与计时（图像 VAE） |
| `trace_video_vae_convs.py` | 视频 VAE 的逐层计时，区分 T=1 无 cache、T>1/cache、普通 Conv2d 与空间 kernel=1 |
| `verify_video_vae_patch.py` | Wan 17 / 81 帧数值、完整 encode 计时和 conv3d 后端验证 |
| `overlay_aiter.py`、`aiter_overlay/` | Lumen 所需、来自其 aiter fork 的 9 个文件 |
| `qwen_image_1024.yaml` | **默认配置**，1024×1024 + `max_sequence_length` 512 |
| `make_data.py`、`qwen_image_smoke.yaml` | smoke 数据集，及 256 配置（`RES=256`） |

### 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `import lumen` 时 `aiter.ops.triton` 下 `ModuleNotFoundError` | 未执行 aiter 补齐 | `python3 $EX/overlay_aiter.py --apply` |
| `could not find path component of main program: 'ld.lld'` | 使用了镜像自带的 flydsl 0.1.6 | 将 `$FLYDSL_SIDECAR` 置于 `PYTHONPATH` 首位 |
| `cannot import name 'fly_values'` | aiter 与 flydsl 0.3.2 不配套 | 预期内且无害，VeOmni 从不导入 aiter |
| flydsl 运行中无 `backend for conv2d: FLYDSL` | VAE 为 FP32 | 使用 `flydsl` 模式，而非单独的 `LUMEN_PATCH=vae_conv` |
| `IndexError: index 0 is out of bounds for dimension 0 with size 0` | BF16 VAE 未对 `encode()` 做 FP32 还原 | 使用 `run_10steps.sh`，其中已正确设置 `vae_bf16` |
| `Conflicting visibility of agent-N` | 两个可见性变量只设了一个 | `HIP_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 需同时设置 |
| 实际步数少于设定，但退出码为 0 | JSONL 记录条数不足 | 需 ≥ `步数 × 卡数`，用 `make_data.py` 重新生成 |
| 宿主机无法删除 `__pycache__` | 容器内以 root 写出 | `docker exec lumen-flydsl-demo rm -rf <路径>` |
| `FATAL: run_10steps.sh is already running as pid N` | 已有运行在进行中 | 等待结束。两个 8 卡运行争抢同组 GPU 会同时污染双方结果。确认该 pid 已不存在后方可删除 `$LOG_DIR/.run.lock` |
| 基线日志中出现 `[lumen]` patch 行 | 两次运行重叠，同时写入 `$VEOMNI_DIR/log.txt` | 串行重跑，上述锁已可防止 |

### 后续工作

按优先级排列：

1. **扩大 Wan 性能测量的样本量，并控制冷启动与运行漂移。** T>1/cache 路径已完成，
   见[第七节](#wan-results)。当前训练与离线 embedding 的重复区间仍重叠，
   需更多重复和更长的稳态窗口才能判断 FlyDSL 对步时的独立贡献。
2. **使 FlyDSL conv 支持 FP32。** 可彻底消除 `vae_bf16` 带来的复杂度，
   使内核在该 trainer 的现有配置下直接可用。
3. **继续评估批量编码与 VAE-only 负载。** Wan 的 `offline_embedding` 已做三次重复，
   内核独立贡献仍未超出散布；它还包含文本编码、数据处理和 embedding 写出，
   不能将完整 VAE encode 的加速比直接套到整步。
4. **使整段 VAE 保持 channels-last。** NHWC 较 NCHW 再快约 38 %
   （6.115 → 3.778 ms），前提是转置开销不按层重复支付。
5. **将 conv3d→conv2d 重写提交至 diffusers 上游。** 对所有 T=1（图像）用户
   均有收益，不依赖 FlyDSL，不改变数值。
6. **实现反向 kernel**，若将来需要训练 VAE。当前 VAE 冻结，非阻塞项。

---

<a id="wan-results"></a>

## 七、Wan2.1：T>1 视频、8 卡训练与离线 embedding

本节数据来自 2026-09-08 的已完成运行，2026-09-09 核对原始日志。
环境沿用上述 ROCm 镜像、flydsl 0.3.2 与 VeOmni commit
`573848a00fcd7329c2411346c6f4a983e9f67e3f`，8× MI355X。
VeOmni 源码未修改；Wan 扩展在本例脚本内实现。

### 1. 测试范围与实际输入

| 测试 | 配置 | 完成情况 |
|---|---|---|
| 单卡卷积 trace / VAE encode | `AutoencoderKLWan`，BF16，17 与 81 帧，480×832 | 逐层计时、完整 encode 与 FP32 参照数值验证通过；encode 各取 5 次内部重复 |
| FP32 回退检查 | 17 帧，480×832 | FlyDSL 后端 cache 为空，确认 FP32 不会启用内核 |
| `online_training` | `Wan2.1-T2V-1.3B` 全参数 DiT SFT，FSDP2，三模式各 3 次 × 10 步 | 9/9 exit 0，均完成 10 步 |
| `offline_embedding` | 同一 condition model、数据、8 个 rank，三模式各 3 次 × 10 步 | 9/9 exit 0，均完成 10 步；不创建 DiT / optimizer，不做 backward |
| `LUMEN_TIME_VAE=1` | 两种任务 × 三模式，各 1 次 × 10 步 | 6/6 exit 0，单独诊断 encode 时间 |

两种任务均使用 400 条 Tom-and-Jerry parquet 数据、81 帧、global batch 8、
每卡 batch 1，实测 VAE 输入为 **`(1, 3, 81, 368, 544)`**。
这与单卡 benchmark 的 **480×832** 不同，不能混用计时。
采用 `wan_video.yaml`：全参数 SFT（不是 LoRA）、eager attention / RoPE、
FSDP2 mixed precision 开启、gradient checkpointing 开启、torch compile 关闭，
所有模型 checkpoint 保存关闭。离线 embedding 会写出 embedding 数据。

### 2. 视频 patch 与内核执行证据

`vae_conv_video` 保留模块原有的 causal padding 和 feature cache 拼接，
通过改绑 `_conv_forward` 替换其底层卷积。仅在 **T=1 且无 cache** 时沿用图像的
conv3d→conv2d 恒等式；带 cache 的 T=1 也必须保留真正的 3D 卷积。
共接管 58 个卷积模块，跳过 13 个空间 kernel=1 的模块。
训练和离线 embedding 的全部 FlyDSL 正式运行均包含：

```text
[lumen] vae_conv_video: patched 58 convolutions, skipped 13
[lumen] backend for conv2d: FLYDSL
[lumen] backend for conv3d: FLYDSL
```

VeOmni 同样以 FP32 加载 Wan VAE；因此 `flydsl` 模式需要
`LUMEN_PATCH=vae_bf16,vae_conv_video`。`vae_bf16` 在 VAE 内部使用 BF16，
并将 encode 的输出恢复为 FP32。必须用 `bf16-only` 对照，才能区分 dtype 与内核收益。

### 3. 单卡：卷积、完整 encode 与数值

以下均为 BF16、480×832；卷积合计是孤立算子计时，完整 encode 是另一次直接计时。

| 指标 | 17 帧 | 81 帧 |
|---|---|---|
| T>1 或带 cache 占原生卷积时间 | 83.3 % | 91.2 % |
| 上述卷积 torch → Lumen | 75.64 → 47.41 ms（1.60×） | 377.97 → 237.23 ms（1.59×） |
| 所有卷积合计 torch → 视频 patch | 90.81 → 54.69 ms（1.66×） | 414.38 → 261.34 ms（1.59×） |
| **完整 VAE encode：原生 → 视频 patch** | **153.2 → 117.1 ms（1.31×）** | **705.8 → 556.1 ms（1.27×）** |
| 原生 BF16 vs FP32（SNR） | 50.2 dB | 49.7 dB |
| 视频 patch BF16 vs FP32（SNR） | 50.4 dB | 50.2 dB |

在本次输入上数值精度未劣化。将 causal padding 错换为对称 padding 的单层反例
仅 **−2.1 dB**，说明验证能检出此类错误；这不是生成质量评估。
旧图像 patch 真正可降维的调用仅占卷积时间 **10.4 % / 2.3 %**（17 / 81 帧）。
此前按时间维形状估算的 21.4 % 覆盖率包含了带 cache 等不可降维调用，应使用本节新统计。

### 4. 正式重复对照：训练与离线 embedding

稳态口径沿用 `compare_runs.py`：根据 wandb 时间戳取 **step 3–10** 的间隔均值，
再汇总三次运行；区间是三次运行均值的 min–max，散布为 `(max−min)/mean`，不是置信区间。
首次调用的 JIT / autotune 不计入稳态，正式矩阵使用已预热的 FlyDSL 磁盘缓存。

| 任务 / 模式 | n | 稳态均值 | 区间 | 散布 | torch 峰值显存 |
|---|---|---|---|---|---|
| 训练：baseline（VAE FP32） | 3 | 4.31 s | 4.24–4.36 s | 2.9 % | 21.19 GB |
| 训练：bf16-only | 3 | 3.42 s | 3.38–3.46 s | 2.2 % | 20.95 GB |
| 训练：BF16 + FlyDSL | 3 | 3.44 s | 3.35–3.50 s | 4.3 % | 21.16 GB |
| 离线 embedding：baseline | 3 | 2.73 s | 2.65–2.78 s | 4.8 % | 14.73 GB |
| 离线 embedding：bf16-only | 3 | 1.82 s | 1.78–1.87 s | 4.6 % | 12.79 GB |
| 离线 embedding：BF16 + FlyDSL | 3 | 1.76 s | 1.65–1.90 s | 14.4 % | 12.85 GB |

**训练中可分辨的收益来自 VAE 的 FP32→BF16：步时约减少 20.6 %。**
再加 FlyDSL 的 3.42→3.44 s 差异小于组内散布，不能据此判断加速或回退。
离线 embedding 的 dtype 收益约为 33 %；再加内核的均值从 1.82→1.76 s，
但区间重叠且 FlyDSL 组散布达 14.4 %，**仍不能主张独立的整步加速**。

训练逐步 loss 相对 baseline-r1 的 mean / max 偏差：baseline 重复为
0.335 % / 0.648 % 和 0.680 % / 3.658 %；bf16-only 为 0.670 % / 2.170 %，
BF16 + FlyDSL 为 0.689 % / 2.354 %。patch 的平均偏差与重复基线相近，
最大偏差低于基线重复中观察到的最大值；10 步不足以证明长期训练等价。
离线 embedding 日志中的 loss / grad_norm 恒为 0，不能作为数值一致性或收敛证据。

### 5. VAE 计时解释了量级，也必须分离冷启动

独立诊断运行对 encode 做 GPU 同步；下表为首次调用之后 9 次的均值。
**不使用这些诊断运行自身的步时做性能对照**，同步会影响调度与重叠。

| 任务 | FP32 encode | BF16 encode | BF16 + FlyDSL encode | 内核节省 |
|---|---|---|---|---|
| 训练 | 1272 ms | 380 ms | 310 ms | 70 ms，约为未插桩 BF16 步时的 2.0 % |
| 离线 embedding | 1268 ms | 381 ms | 307 ms | 74 ms，约为未插桩 BF16 步时的 4.1 % |

训练中 VAE encode 占对应未插桩步时约 **29.5 % / 11.1 % / 9.0 %**。
这些是完整 encode 占比，不能称为“卷积占比”。内核收益与运行散布处于相近量级。
另一次冷缓存探索运行的首步约需 4.5 分钟；正式三重复矩阵不包含该运行。
首次 encode 还会受到 JIT / autotune 影响，不能将其与后续调用混算为稳态均值。

### 6. 结果来源与复查入口

正式结果来自 `compare_online.log` / `compare_embed.log`，对应
`wan-{baseline,bf16-only,flydsl}-r{1,2,3}` 与
`wanemb-{baseline,bf16-only,flydsl}-r{1,2,3}` 的日志、meta 和 offline wandb。
诊断来自 `timevae_online.log` / `timevae_embed.log`；单卡数据来自
`trace_wan_17f_v2.log`、`trace_wan_81f.log`、`verify_wan_bf16.log`、
`verify_wan_bf16_81f.log`、`verify_wan_fp32.log`。
这些是实验产物名，原始日志不随仓库分发。首轮 `discarded-pass1` 在运行期间改过脚本，
不纳入正式性能表。

在前述容器与依赖准备完成、`WAN_DIR` 指向 Wan2.1-T2V-1.3B Diffusers 模型、
`WAN_DATA_DIR` 指向按 VeOmni Wan 数据格式准备的 parquet（至少 80 条）后，
可用 `run_wan_10steps.sh` 串行运行三模式。设 `RUN_SUFFIX=r1` / `r2` / `r3`
保留重复结果，`WAN_TASK=offline_embedding` 切换离线任务。
三模式必须共用 sidecar 环境，baseline 的 `LUMEN_PATCH` 应未设置，正式计时的
`LUMEN_TIME_VAE` 应未设置；单独诊断时才设 `LUMEN_TIME_VAE=1`。
用 `compare_runs.py` 传入全部重复名汇总，勿只选最快的一次。

**范围边界**：本节训练对象是 Wan2.1-T2V-1.3B，未跑 Wan 14B / I2V / LoRA；
同一 VAE 的单卡结果不等于这些模型的训练结果。当前验证针对冻结 VAE 的 encode 前向，
没有验证 VAE 训练反向、decode 性能、长程收敛或视频生成质量。
