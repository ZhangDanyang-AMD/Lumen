# Qwen-Image / Wan2.1 VAE 接入 Lumen FlyDSL 卷积

> 对外集成报告与可复现 Runbook。English version: [README_EN.md](README_EN.md)

本文验证 Lumen FlyDSL implicit-GEMM 卷积在 VeOmni 的两类真实负载中是否正确执行、
数值是否可靠，以及局部算子收益能否传导到端到端：

- Qwen-Image：时间维退化为 `T=1` 的 causal Conv3d，可精确降维为 Conv2d。
- Wan2.1：真实 `T>1` 视频和 feature cache 路径，必须保留完整 causal Conv3d。

全部训练在单节点 8× AMD Instinct MI355X（gfx950）上完成。本文只使用公开仓库、
公开模型标识和通用 `/work` 容器路径，不依赖任何内部集群或个人目录。

---

## 1. 执行摘要

### 1.1 最终结论

| 项目 | Qwen-Image | Wan2.1 |
|---|---|---|
| 内核执行证据 | `backend for conv2d: FLYDSL` | `backend for conv2d: FLYDSL` 与 `backend for conv3d: FLYDSL` |
| 局部卷积收益 | 1024 下 NCHW 3.07×，NHWC 4.97× | 81 帧 T>1/cache 卷积 1.59× |
| 完整 VAE encode | `33.05 → 17.22 ms`，**1.92×** | `705.8 → 556.1 ms`，**1.27×** |
| 训练中可优化部分 | 完整 encode 约占 step 0.82%；卷积约占 0.56% | BF16 encode 约占 step 11.1% |
| 理论训练 step 收益 | 约 **0.40%** | 约 **2.05%** |
| 端到端结论 | 低于 ±6% 重复运行噪声，不能主张训练加速 | 低于 2.2–4.3% 组内散布，不能主张训练加速 |

可以明确成立的结论：

1. FlyDSL 内核确实进入了 8 卡训练和离线 embedding 的执行路径，不是静默回退。
2. Qwen-Image 和 Wan 的 BF16 encode 精度相对 FP32 均未劣化。
3. 完整 VAE encode 分别获得 1.92× 和 1.27× 加速。
4. 当前训练配置下，FlyDSL 的独立 step-time 收益小于运行噪声，不能据单次 A/B 报加速比。
5. Wan 中稳定可见的 `4.31 → 3.42 s/step`（−20.6%）来自 VAE FP32→BF16，
   **不是 FlyDSL 内核**；因此必须保留 `bf16-only` 对照。

### 1.2 为什么算子提升很大，训练提升却很小

端到端收益受 Amdahl 定律限制：

```text
端到端节省比例 = 被优化部分占比 × (1 - 1 / 局部加速比)
```

| 场景 | 被优化部分 | 局部变化 | 占 step | 理论 step 节省 |
|---|---|---:|---:|---:|
| Qwen 256 | VAE 卷积 | `2.771 → 1.007 ms` | 0.068% | **0.043%** |
| Qwen 1024 | 完整 VAE encode | `33.05 → 17.22 ms` | 0.824% | **0.395%** |
| Wan 训练 | BF16 VAE encode | `380 → 310 ms` | 11.1% | **2.047%** |
| Wan offline embedding | BF16 VAE encode | `381 → 307 ms` | 20.9% | **4.066%** |

Qwen 的局部加速主要来自一次**算法级降维**：causal Conv3d 在 `T=1` 时只有最后一个
时间 kernel slice 参与有效计算，因此可精确改写为 Conv2d。这同时去掉了无效零帧计算，
并避开 torch Conv3d 对退化 5D shape 的低效路径，所以单层可达到 2–3.7×。
但 Qwen 的冻结 VAE 每步只前向一次，后面仍有 20.43B DiT 的 forward、backward、
optimizer 和 FSDP 通信，最终只能省约 16 ms/step。

Wan 不能做同样的降维：后续 chunk 的 feature cache 是真实特征而不是零。
这里的 1.59× 是真实 Conv3d kernel 收益。81 帧下卷积占完整 encode 的约 58.7%，
因此按 Amdahl 定律，卷积 `414.38 → 261.34 ms` 应把 encode 降到约 553 ms，
实测为 556.1 ms；局部收益没有丢失，只是被 VAE 的非卷积部分稀释为 1.27×。

进一步进入训练后，Wan BF16 VAE 每步省 70 ms，而 step 为 3.42 s，理论收益只有 2.05%。
这一量级与 2.2–4.3% 的重复运行散布重叠，所以观测到 `3.42 vs 3.44 s` 既不能说明
回退，也不能说明加速。离线 embedding 去掉 DiT 和 backward 后理论收益升到约 4.1%，
但三次 FlyDSL 运行的散布为 14.4%，仍不足以下结论。

### 1.3 从算子到端到端，收益经过四层稀释

1. **只替换合适的卷积。** Pointwise、部分 resampler 和不支持的 dtype 会保留在 torch。
2. **卷积只是 VAE 的一部分。** Padding、cache 拼接、normalization、activation、
   resampling 和 latent distribution 不会因卷积 kernel 加速而消失。
3. **VAE 只是 step 的一部分。** DiT 前后向、优化器、FSDP/RCCL、文本编码、数据处理和
   写盘占据其余时间。
4. **短任务还要支付 JIT。** Qwen 首次编译约 10–11 秒；Wan 冷缓存约 4.5 分钟。
   按 Wan 每步节省 70 ms 计算，冷启动约需 3,850 步才能摊平。

### 1.4 报告边界

本文可以引用算子、完整 encode、数值正确性和后端执行证据；不能引用稳定的训练加速比。
Qwen 的 500-step 运行使用 8 张循环图片，只证明训练链路可收敛并发生过拟合，不代表泛化。
Wan 尚未测试长程收敛、生成质量、14B、I2V、LoRA、VAE decode 或 VAE 训练反向。

---

## 2. 测试设计

### 2.1 固定环境

| 项 | 配置 |
|---|---|
| GPU | 单节点 8× AMD Instinct MI355X，gfx950 |
| 容器 | `amdagi/veomni:rocm7.14_torch2.12_py3.12` |
| VeOmni | `573848a00fcd7329c2411346c6f4a983e9f67e3f`，源码零修改 |
| FlyDSL | 0.3.2，安装在独立 sidecar 目录 |
| 并行 | FSDP2，8 ranks，micro batch 1 |
| 共同设置 | eager attention / RoPE、gradient checkpointing、`torch_compile=false` |

### 2.2 已完成测试矩阵

| 模型 / 任务 | 测试 |
|---|---|
| Qwen-Image 基线 | 256×256 全参数 SFT：2 / 10 / 500 steps，全部 exit 0 |
| Qwen-Image + Lumen | 256 下 8 次 10-step 变体与重复；1024 下独立复现、算子验证和 10-step A/B |
| Wan 单卡 | 17 / 81 帧逐层 trace、完整 encode、FP32/BF16 数值与后端检查 |
| Wan 训练 | 3 模式 × 3 次重复 × 10 steps，共 9 次，全部 exit 0 |
| Wan offline embedding | 3 模式 × 3 次重复 × 10 steps，共 9 次，全部 exit 0 |
| Wan VAE 诊断 | 两种任务 × 3 模式，各 1 次 10 steps，共 6 次 |

### 2.3 三模式归因

| 模式 | VAE dtype | 卷积实现 | 用途 |
|---|---|---|---|
| `baseline` | FP32 | torch | VeOmni 当前行为 |
| `bf16-only` | BF16，encode 输出恢复 FP32 | torch | 单独测量 dtype 收益 |
| `flydsl` | BF16，encode 输出恢复 FP32 | Lumen FlyDSL | 在相同 dtype 下测内核增量 |

FlyDSL 当前仅支持 BF16；若直接对比 FP32 baseline 和 BF16+FlyDSL，会把 dtype 收益错误地
记在内核名下。所有模式使用相同的 sidecar 环境、配置、数据、seed 和并行度。

### 2.4 测量口径

- 后端执行以日志中的 `backend for conv2d/conv3d: FLYDSL` 为唯一证据。
- 数值以同一输入的 FP32 encode 为参照，使用 SNR；不要求 BF16 逐位相同。
- 训练稳态取 wandb 时间戳的 step 3–10 间隔，step 1 的 JIT/autotune 不计入。
- 性能结论比较模式间差距与模式内重复散布；小于散布的差异不视为结果。
- `LUMEN_TIME_VAE=1` 会逐次同步 GPU，只用于测 encode，不使用该 run 的 step time。

---

## 3. 实测结果

### 3.1 Qwen-Image

#### 数值与后端

```text
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

| 对照项 | SNR |
|---|---:|
| 原生 BF16 vs FP32 | 51.6 dB |
| **接入后 BF16 vs FP32** | **51.7 dB** |
| 单层 conv3d→conv2d 重写 | 90.4 dB |
| 单层 torch conv2d vs Lumen conv2d | 51.0 dB |

#### 性能与占比

| 指标 | 256×256 | 1024×1024 |
|---|---:|---:|
| 完整 VAE encode，原生→接入 | `4.44 → 3.83 ms`（1.16×） | `33.05 → 17.22 ms`（**1.92×**） |
| 仅卷积，NCHW / NHWC | 2.23× / 2.67× | **3.07× / 4.97×** |
| 卷积占训练 step | 0.068% | 0.56% |
| baseline 稳态 step | 3.95 s | 4.01 s |

256 下三次 baseline 与三次 patched 的稳态均值为 4.06 与 4.04 s，区间完全重叠。
1024 的单次 A/B 为 4.01 与 3.91 s，但理论上限只有约 0.4%，因此这 2.5% 差异不能归因
于内核。相同配置的重复波动约 ±6%。

Qwen 基线 500 steps 的 mean loss 从 `0.02083` 降至 `0.00621`（−70.2%），
500/500 均为有限值，无 OOM、hang 或显存持续增长。该结果是优化链路与过拟合验证。

### 3.2 Wan2.1 单卡 VAE

`vae_conv_video` 接管 58 个卷积模块，跳过 13 个空间 kernel 为 1 的模块：

```text
[lumen] vae_conv_video: patched 58 convolutions, skipped 13
[lumen] backend for conv2d: FLYDSL
[lumen] backend for conv3d: FLYDSL
```

以下均为 BF16、480×832；完整 encode 各取 5 次内部重复：

| 指标 | 17 帧 | 81 帧 |
|---|---:|---:|
| T>1 或带 cache 占原生卷积时间 | 83.3% | 91.2% |
| T>1/cache 卷积，torch→Lumen | `75.64 → 47.41 ms`（1.60×） | `377.97 → 237.23 ms`（1.59×） |
| 所有卷积，原生→视频 patch | `90.81 → 54.69 ms`（1.66×） | `414.38 → 261.34 ms`（1.59×） |
| **完整 VAE encode** | **`153.2 → 117.1 ms`（1.31×）** | **`705.8 → 556.1 ms`（1.27×）** |
| 原生 BF16 vs FP32 | 50.2 dB | 49.7 dB |
| 视频 patch BF16 vs FP32 | 50.4 dB | 50.2 dB |

把 causal padding 错换成对称 `padding=1` 的单层反例只有 −2.1 dB，说明验证能够检出
时间语义错误。旧的图像 patch 真正能降维的调用只占 17/81 帧卷积时间的 10.4%/2.3%。

### 3.3 Wan 8 卡训练与 offline embedding

训练对象为 Wan2.1-T2V-1.3B 全参数 SFT，81 帧，实测 VAE 输入
`(1, 3, 81, 368, 544)`。训练与 offline embedding 各完成三模式、每模式三次重复。

| 任务 / 模式 | n | 稳态均值 | 三次区间 | 散布 | torch 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 训练：baseline（FP32） | 3 | 4.31 s | 4.24–4.36 s | 2.9% | 21.19 GB |
| 训练：bf16-only | 3 | 3.42 s | 3.38–3.46 s | 2.2% | 20.95 GB |
| 训练：BF16 + FlyDSL | 3 | 3.44 s | 3.35–3.50 s | 4.3% | 21.16 GB |
| Offline embedding：baseline | 3 | 2.73 s | 2.65–2.78 s | 4.8% | 14.73 GB |
| Offline embedding：bf16-only | 3 | 1.82 s | 1.78–1.87 s | 4.6% | 12.79 GB |
| Offline embedding：BF16 + FlyDSL | 3 | 1.76 s | 1.65–1.90 s | 14.4% | 12.85 GB |

训练中 baseline→bf16-only 的 −20.6% 明确超出散布；bf16-only→FlyDSL 的 +0.6%
位于散布内，不构成回退。Offline embedding 的均值改善约 3.3%，但区间重叠，仍不能
主张稳定加速。

训练逐步 loss 相对 baseline-r1 的 mean/max 偏差：baseline 重复为
0.335%/0.648% 与 0.680%/3.658%，bf16-only 为 0.670%/2.170%，
BF16+FlyDSL 为 0.689%/2.354%。patch 偏差没有超过重复基线的观测范围；
10 steps 不足以证明长期训练等价。Offline embedding 的 loss/grad norm 恒为 0，
不能作为数值一致性或收敛证据。

### 3.4 Wan VAE 计时与冷启动

以下是同步计时诊断中排除首次调用后的 9 次均值：

| 任务 | FP32 encode | BF16 encode | BF16 + FlyDSL | 内核节省 |
|---|---:|---:|---:|---:|
| 训练 | 1272 ms | 380 ms | 310 ms | 70 ms，约占 BF16 step 2.0% |
| Offline embedding | 1268 ms | 381 ms | 307 ms | 74 ms，约占 BF16 step 4.1% |

视频 VAE 有约 66 类不同卷积 shape。冷缓存首次 FlyDSL run 的 step 1 约需 4.5 分钟编译；
缓存热后 step 1 只比 baseline 多约 1–2 秒。正式性能矩阵使用热缓存并排除 step 1。

---

## 4. 实现说明

### 4.1 Qwen：T=1 精确降维

`QwenImageCausalConv3d` 在时间维前补 `kT-1` 个零帧、后不补，再以 `padding=0` 卷积。
当输入只有一个真实帧时：

```text
conv3d(causal_pad(x), w) == conv2d(x, w[:, :, -1])
```

这是精确恒等式，不是近似。Patch 只重新绑定 `forward`，不修改参数和模块树，
因此兼容 `init_device=meta` 和 FSDP2。带 cache、`T>1` 或时间 stride/dilation 不满足
恒等式时，不走该路径。

### 4.2 Wan：保留 causal/cache，只替换底层 kernel

Wan 的后续 chunk 会把 feature cache 拼入输入，不能丢弃其他时间 kernel slice。
`vae_conv_video` 不重新实现 causal 逻辑，而是重新绑定 `nn.Conv2d/3d._conv_forward`：

```python
def _conv_forward(self, input, weight, bias):
    return lumen_conv(
        input,
        weight,
        bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups,
    )
```

模块原有的 padding、cache 拼接和 chunk 逻辑保持不变。空间 kernel 为 1 的 pointwise
卷积在实测中约为 0.99–1.00×，继续使用 torch。

### 4.3 BF16 约束

VeOmni 当前以 FP32 加载 Qwen 与 Wan VAE，而 FlyDSL conv 只支持 BF16。FP32 调用会在
进入 dispatcher 前静默回退，backend cache 为空。因此 `flydsl` 模式组合：

```text
vae_bf16 + vae_conv          # Qwen
vae_bf16 + vae_conv_video    # Wan
```

单纯 `vae.to(bfloat16)` 会让下游 diffusion timestep 在 FP32 schedule 中查找失败。
本例在 VAE 内部使用 BF16，并在 `encode()` 出口把 latents 恢复为 FP32，使后续 dtype
与 baseline 一致。支持 FP32 conv 是消除该复杂度的根本方案。

### 4.4 Sidecar 环境

镜像内置 flydsl 0.1.6；本例把 0.3.2 安装到独立目录，仅通过 `PYTHONPATH` 启用，
不覆盖镜像环境。`overlay_aiter.py` 只补齐 Lumen import 所需且当前不存在的 9 个文件，
不会覆盖 aiter 已有文件。所有 A/B 模式使用相同 sidecar 环境。

---

## 5. Runbook

### 5.1 准备容器

需要单节点 8× MI355X、ROCm 驱动，以及约 200 GB 可用空间（同时下载两个模型时）。
将下面的 `ROOT` 替换为任意高速本地磁盘目录：

```bash
export ROOT=/path/to/fast-storage/lumen-flydsl-demo
mkdir -p "$ROOT"

docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12

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
  -v "$ROOT":/work -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity

dx() { docker exec -w /work lumen-flydsl-demo bash -lc "$*"; }
dx 'rocm-smi --showid | head -12'
```

`HIP_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 必须同时设置。

### 5.2 获取代码并配置环境

```bash
dx 'git clone -b dev/flydsl-conv3d https://github.com/ZhangDanyang-AMD/Lumen.git /work/Lumen'

dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    cd /work/VeOmni &&
    git checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f'

export EX=/work/Lumen/examples/qwen-image-flydsl-conv
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

期望输出包含：

```text
torch.cuda  None (must be None on ROCm)
devices     8
flydsl conv3d probe: True (must be True)
SETUP OK
```

不要执行 `uv sync`、`pip install -e '.[gpu]'`，也不要安装 CUDA/NVIDIA wheel；
不要设置 `HSA_OVERRIDE_GFX_VERSION`。

### 5.3 Qwen-Image：验证与训练

```bash
# 模型和 10-step 数据
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"

# 算子、真实卷积 trace、完整 VAE patch
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_lumen_conv.py"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/trace_vae_convs.py"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype bf16"

# 快速功能验证：各一次
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py baseline-1024 flydsl-1024"
```

若要比较性能，至少做三次重复；脚本默认串行锁定，避免两个 8 卡任务重叠：

```bash
for r in r1 r2 r3; do
  dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_10steps.sh baseline"
  dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  baseline-1024-r1 baseline-1024-r2 baseline-1024-r3 \
  flydsl-1024-r1 flydsl-1024-r2 flydsl-1024-r3"
```

使用 `RES=256` 可切换到小分辨率。数据条数必须不少于 `steps × ranks`；脚本会自动检查。

### 5.4 Wan：模型与数据

```bash
dx ". $EX/env.sh && hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir \$WAN_DIR"

# 下载公开数据集的 manifests 与前 400 个视频条目
dx '. /work/Lumen/examples/qwen-image-flydsl-conv/env.sh &&
    RAW=$WORK/data/tom-and-jerry &&
    mkdir -p "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include captions.txt --local-dir "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include videos.txt --local-dir "$RAW" &&
    head -n 400 "$RAW/videos.txt" > "$RAW/videos.head.txt" &&
    xargs -a "$RAW/videos.head.txt" -n 25 \
      hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset \
      --repo-type dataset --local-dir "$RAW" --quiet'
```

VeOmni 的转换器要求 `captions.txt` 与 `videos.txt` 一一对应。根据已下载成功的视频生成
子集 manifest，然后转换为 parquet：

```bash
docker exec -i -w /work lumen-flydsl-demo python3 - <<'PY'
from pathlib import Path

src = Path('/work/data/tom-and-jerry')
dst = Path('/work/data/tom-and-jerry-subset')
dst.mkdir(parents=True, exist_ok=True)
link = dst / 'videos'
if not link.exists():
    link.symlink_to(src / 'videos', target_is_directory=True)
videos = src.joinpath('videos.txt').read_text().splitlines()
captions = src.joinpath('captions.txt').read_text().splitlines()
keep = [(v, c) for v, c in list(zip(videos, captions))[:400] if (src / v).exists()]
dst.joinpath('videos.txt').write_text(''.join(f'{v}\n' for v, _ in keep))
dst.joinpath('captions.txt').write_text(''.join(f'{c}\n' for _, c in keep))
print(f'{len(keep)} clips kept')
assert len(keep) >= 80
PY

dx ". $EX/env.sh && python3 \$VEOMNI_DIR/scripts/multimodal/convert_data/tom-and-jerry.py \
  --dataset_path \$WORK/data/tom-and-jerry-subset \
  --output_dir \$WAN_DATA_DIR"
```

`WAN_DATA_DIR` 应包含至少 80 行 parquet；建议使用 400 行以保留余量。

### 5.5 Wan：单卡验证

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/trace_video_vae_convs.py --model \$WAN_DIR --frames 17"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/trace_video_vae_convs.py --model \$WAN_DIR --frames 81"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/verify_video_vae_patch.py --model \$WAN_DIR --frames 17 --dtype bf16"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/verify_video_vae_patch.py --model \$WAN_DIR --frames 81 --dtype bf16"
```

最后一项应以 `PASS` 结束，并显示 conv2d、conv3d 均锁定到 `FLYDSL`。
用 `--dtype fp32` 可验证预期的静默回退：backend cache 应为空。

### 5.6 Wan：训练与 offline embedding

快速功能验证：

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh bf16-only"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh flydsl"

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wan-baseline wan-bf16-only wan-flydsl"
```

正式性能对照必须重复：

```bash
for r in r1 r2 r3; do
  for mode in baseline bf16-only flydsl; do
    dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_wan_10steps.sh $mode"
  done
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wan-baseline-r1 wan-baseline-r2 wan-baseline-r3 \
  wan-bf16-only-r1 wan-bf16-only-r2 wan-bf16-only-r3 \
  wan-flydsl-r1 wan-flydsl-r2 wan-flydsl-r3"
```

切换到 offline embedding：

```bash
for r in r1 r2 r3; do
  for mode in baseline bf16-only flydsl; do
    dx ". $EX/env.sh && WAN_TASK=offline_embedding RUN_SUFFIX=$r \
      bash \$EXAMPLE_DIR/run_wan_10steps.sh $mode"
  done
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wanemb-baseline-r1 wanemb-baseline-r2 wanemb-baseline-r3 \
  wanemb-bf16-only-r1 wanemb-bf16-only-r2 wanemb-bf16-only-r3 \
  wanemb-flydsl-r1 wanemb-flydsl-r2 wanemb-flydsl-r3"
```

单独测量 VAE encode 时设置 `LUMEN_TIME_VAE=1`；不要把该运行的 step time 纳入正式对比。

### 5.7 验收标准

- 所有训练日志必须完成目标 steps 且退出码为 0。
- FlyDSL 模式必须打印对应的 `backend ...: FLYDSL`；仅看到 `patched` 不足以证明内核运行。
- BF16 数值 SNR 应与原生 BF16 相对 FP32 的量级一致。
- 性能结论必须基于重复分布；模式间差距小于组内散布时报告“无法分辨”。
- Step 1 与冷缓存编译单独报告，不与稳态均值混合。

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `import lumen` 时缺少 `aiter.ops.triton` 模块 | 未执行 aiter overlay | 重新运行 `setup_env.sh` |
| `ld.lld` 路径错误 | 使用了镜像自带 flydsl 0.1.6 | 确认 `$FLYDSL_SIDECAR` 位于 `PYTHONPATH` 首位 |
| `cannot import name 'fly_values'` | aiter 与 sidecar flydsl API 不一致 | 本例预期内；VeOmni 不依赖 aiter CK/HIP ops |
| 无 `backend for conv*: FLYDSL` | FP32、不支持的 shape，或 sidecar 未生效 | 检查 dtype、`PYTHONPATH` 和 backend cache |
| BF16 VAE 后 timestep 查找 `IndexError` | encode 输出未恢复 FP32 | 使用本例入口与 `vae_bf16` 模式 |
| 实际 steps 少于目标但 exit 0 | 数据行数不足 | 准备至少 `steps × ranks` 条记录 |
| `Conflicting visibility of agent-N` | 两个 GPU 可见性变量不一致 | 同时设置 HIP 与 CUDA visibility |
| baseline 日志出现 patch 行 | 两个运行重叠或继承了错误环境 | 串行运行；脚本会清除 baseline 的 `LUMEN_PATCH` |
| Wan 首次运行长时间停在 step 1 | 多 shape 的 FlyDSL JIT | 等待编译完成；保留并复用磁盘 cache |

---

## 7. 文件索引与后续工作

| 文件 | 作用 |
|---|---|
| `lumen_vae_conv.py` | T=1 精确降维、T>1/cache kernel 替换与层选择 |
| `train_dit_lumen.py` | VeOmni hook、BF16 VAE、后端与 VAE 计时报告 |
| `run_10steps.sh` | Qwen 三模式训练，支持 `RES` 与 `RUN_SUFFIX` |
| `run_wan_10steps.sh` / `wan_video.yaml` | Wan 训练和 offline embedding 三模式入口 |
| `verify_vae_patch.py` | Qwen 数值、完整 encode 和后端验证 |
| `verify_video_vae_patch.py` | Wan 17/81 帧数值、encode 和 conv3d 后端验证 |
| `trace_vae_convs.py` / `trace_video_vae_convs.py` | 逐层 shape、覆盖率和性能归因 |
| `compare_runs.py` | 多次运行的 loss、稳态步时、散布和显存汇总 |
| `survey_veomni_convs.py` | VeOmni 架构与 VAE 卷积覆盖普查 |

下一步优先级：

1. 增加 Wan 重复次数和稳态长度，用成对交错 A/B 控制时间漂移。
2. 为 FlyDSL conv 增加 FP32 支持，移除 `vae_bf16` 这一混杂变量。
3. 测试批量 VAE encode、VAE-only 服务等真正由卷积主导的负载。
4. 让整个 VAE 阶段保持 channels-last，避免逐层布局转换。
5. 将 Qwen 的 T=1 Conv3d→Conv2d 精确重写贡献到 diffusers 上游。
6. 若未来训练 VAE，再实现 dgrad/wgrad；当前冻结 VAE 不需要反向 kernel。
