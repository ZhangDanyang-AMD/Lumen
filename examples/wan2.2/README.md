# SimWAM Wan2.2 VAE（NAVSIM GRPO）：torch BF16 与 aiter FlyDSL 卷积

> English version: [README_EN.md](README_EN.md)

SimWAM 的 Wan2.2-TI2V-5B VAE（`WanVideoVAE38`）在 NAVSIM GRPO 场景下的卷积性能报告：原生 torch BF16 对比 aiter 的
FlyDSL implicit-GEMM 卷积。FlyDSL 接入代码 `flydsl_conv.py` 只放在本示例目录里，Lumen 库本身没有改动。

GRPO 场景是 FlowGRPO rollout 的条件编码：每个样本只把当前帧编码一次，不解码（`trainer_grpo.py`、
`simwam_grpo.build_action_condition`）。navtest 的 action-only eval 也用同样的方式编码同一帧。所以本文只测这一种调用：

| | shape |
|---|---|
| 输入 | (1, 3, 1, 384, 672)，BF16 |
| latent | (1, 48, 1, 24, 42) |
| decode | 无 |
| 每次编码的卷积 | 30 次调用，16 种不同的卷积（11 种 Conv3d、5 种 Conv2d） |

---

## 1. 结论

| 项 | torch BF16 | aiter FlyDSL |
|---|---:|---:|
| 单帧编码（`cudnn.deterministic=False`，两轮中位数取平均，host 低负载） | 7.372 ms | **5.404 ms（1.36×）** |
| 单帧编码（torch 开 `cudnn.deterministic=True`，即 SimWAM eval 的设置） | 15.708 ms | **5.404 ms（2.91×）** |
| 每次编码的 GPU kernel 时间（torch.profiler） | 7.116 ms | 3.101 ms |
| 其中卷积 kernel | 3.816 ms | 1.234 ms（3.09×） |
| 卷积调用：aiter / torch | 0 / 30 | 30 / 0 |
| 调优表查表：精确 / 借用 / 启发式 | – | 25 / 0 / 0 |
| 16 种卷积的三值逐位检查 | – | 16 / 16 逐位相同 |
| latent 对 FP32 的 SNR | 44.6 dB | 45.0 dB |
| 完整 navtest PDMS（12146 个 token） | 90.27 | 90.30（差 +0.03，p = 0.48，不显著） |

- FlyDSL 路径比 torch BF16 快 1.36 倍；对 SimWAM eval 实际使用的 `cudnn.deterministic=True` 快 2.91 倍。
- FlyDSL 的编码受 host 端限制：GPU 上只有 3.10 ms 的 kernel，墙钟却是 5.40 ms。所以它的耗时会随 host 的 CPU 负载变化，
  同机跑着 6 卡 navtest 时测到 5.82–6.00 ms；torch 的编码受 GPU 限制，几乎不受影响（§2.1）。
- 精度没有回退：每种卷积的 kernel 都逐位正确，latent 离 FP32 参考不比 torch BF16 远，完整 navtest 的 PDMS 和 torch BF16
  分不出差别。和 torch BF16 按 rtol = atol = 1e-5 比较时有 55.9% 的元素超出，最大差 0.031，这是两边求和顺序不同造成的
  BF16 舍入差异，见 §2.3。
- 在 GRPO 训练里 VAE 只占整步约 0.4%。实测换成 FlyDSL 后每步约省 0.4 ms，约占整步 0.07%（2 卡、20 步的粗测，§2.5）。
- 25 次查表全部精确命中，前提是装了 [ROCm/aiter#7372](https://github.com/ROCm/aiter/pull/7372) 的 Wan2.2 调优表；
  没有这张表时查表全部走启发式，编码变慢，日志里有警告（§2.4）。

适用范围：AMD Instinct MI350X（gfx950），ROCm 7.14 / torch 2.12 镜像 `amdagi/veomni:rocm7.14_torch2.12_py3.12`，aiter main
`78a14ae5d` 加 #7372 `db4c8d83` 的 Wan2.2 表，flydsl 0.5.0，SimWAM `1433ee7`，`Wan2.2_VAE.pth`。测于 2026-11-17。

---

## 2. 数据

### 2.1 单帧编码耗时

同卡 A/B：每一轮在同一张卡、同一个进程里依次测三种方案，输入是同一个固定种子的 BF16 帧；每种方案 warmup 10 次后计时
100 次，每次前后都同步，取中位数。第 2 轮换一张卡，顺序反过来，最后对两轮的中位数取平均。两轮期间卡上没有别人的进程，
host 的 1 分钟 load average 在 40–75 之间（384 个逻辑核）。命令见 §3.3（`bench_encode.py`）。

| 方案 | 第 1 轮（GPU 2） | 第 2 轮（GPU 4） | 平均 | 相对 torch BF16 |
|---|---:|---:|---:|---:|
| torch BF16，`cudnn.deterministic=False` | 7.344 ms | 7.400 ms | 7.372 ms | 1.00 |
| torch BF16，`cudnn.deterministic=True` | 15.733 ms | 15.682 ms | 15.708 ms | 0.47 |
| aiter FlyDSL（本示例 `flydsl_conv.py`） | 5.386 ms | 5.421 ms | **5.404 ms** | **1.36** |

第 1 轮的顺序是 torch → torch（deterministic）→ FlyDSL，第 2 轮是 FlyDSL → torch（deterministic）→ torch。变异系数都在
1.2% 以内。

**host 负载的影响**：同样的两轮在同机跑着 6 卡 navtest 时（load average 约 500）测过一次，torch 是 7.359 / 7.329 ms，
FlyDSL 是 5.997 / 5.820 ms（平均 5.909 ms，1.24×），FlyDSL 的变异系数 3.3% / 3.9%，分布有两个峰（第 1 轮 p10 是 5.557 ms）。
torch 两次测量相差不到 1%，FlyDSL 差 9%，原因是 FlyDSL 编码受 host 端限制（§2.2）。同一进程里交替测本示例的补丁和 Lumen
`ccaeadc` 的 `lumen_vae_conv.py`（SimWAM 评测用的补丁），两者在 load average 450 时是 5.390 / 5.367 ms，在 64 时是
5.412 / 5.379 ms，latent 逐位相同：差别来自 host 负载，不是两个补丁的代码路径。

### 2.2 kernel 级证据

`bench_encode.py` 最后用 torch.profiler 记录 20 次编码，把每个 GPU kernel 的时间加起来，再按 kernel 名分成卷积和其余两类。
单独给每个卷积计时、前后各同步一次的做法测到的主要是 launch 和 host 的时间（1×1 卷积和 3×3 卷积看起来一样长），所以这里
看 profiler。数据是 §2.1 的两轮，每次编码：

| | torch BF16（第 1 轮 / 第 2 轮） | aiter FlyDSL（第 1 轮 / 第 2 轮） |
|---|---:|---:|
| GPU kernel 总时间 | 7.086 / 7.146 ms | 3.111 / 3.091 ms |
| 卷积 kernel | 3.784 / 3.847 ms（25 次 launch） | 1.238 / 1.230 ms（25 次 launch） |
| 其余 kernel | 3.302 / 3.300 ms | 1.874 / 1.861 ms |
| 墙钟（§2.1） | 7.344 / 7.400 ms | 5.386 / 5.421 ms |

- 卷积 kernel：torch 主要是 CK 的 grouped conv kernel（16 次 2.67 ms、4 次 0.83 ms）；FlyDSL 是 25 次 `conv3d_implicit_kernel`，
  合计 1.23 ms，快 3.09 倍。两边的卷积列都是 25 次 launch，另外 5 个 1×1 卷积的 kernel 名不含卷积关键字，计在其余里。
- 其余 kernel 也少了约 1.43 ms。推断（没有逐项拆分）：stock 路径要在时间维给每个 CausalConv3d 的输入补两帧零，
  再做 kT=3 的卷积；T=1 改写后没有这些补零的张量。
- torch 的 GPU 时间接近墙钟，受 GPU 限制；FlyDSL 的 GPU 时间只有墙钟的 57%，其余是 GPU 在等 host 发 kernel，所以 FlyDSL
  对 host 的 CPU 负载更敏感（§2.1）。
### 2.3 数值

| 检查 | 结果 |
|---|---|
| 调用数 | 每次单帧编码 30 次调用，30 次走 FlyDSL kernel，0 次回退 torch（`test_flydsl_encode_matches_stock`） |
| 三值逐位检查 | FlyDSL 实际执行的 16 种卷积（T=1 改写之后的形式）各跑一次 {-1, 0, 1} 的 BF16 数据，和 CPU fp32 参考 16 / 16 逐位相同（`test_flydsl_kernels_exact_on_grpo_conv_shapes`） |
| 对 FP32 的 SNR | stock torch BF16 44.6 dB，FlyDSL 45.0 dB（`verify_vae.py --backend flydsl`） |
| 对 torch BF16，rtol = atol = 1e-5 | 不满足：55.9% 的 latent 元素超出，最大差 0.031。每层都把 fp32 累加结果舍入成 BF16，求和顺序不同就会差一个 BF16 ulp，后面的层再把它传下去。判据照原样保留，结果照实报告 |
| T=1 改写本身 | CPU 上（全部回退 torch）小模型的单帧编码，改写前后最大差 3.3e-7，满足 1e-5（`test_flydsl_patch_binds_every_conv_and_t1_rewrite_is_exact`） |

**完整 navtest 的 PDMS**（12146 个 token，cv350，2026-11-17）：

| 指标 | torch BF16 | aiter FlyDSL（main + #7372 表） |
|---|---:|---:|
| **PDMS** | **90.27** | **90.30** |
| NC / DAC / EP / TTC | 98.65 / 98.04 / 84.02 / 95.78 | 98.64 / 98.05 / 84.02 / 95.82 |
| 对 GT 的 ADE | 0.5798 m | 0.5799 m |

- 配对比较：差 +0.025，bootstrap 95% CI −0.045 到 +0.097，配对 t 检验 p = 0.48，不显著。38 个 token 有 gate 翻转（有利 21、
  不利 15，符号检验 p = 0.41）。对 torch BF16 的 action SNR 中位数 52.2 dB。12146 个 token 每个 30 次卷积全部走 FlyDSL。
- 和没有 #7372 表的 FlyDSL（aiter main，查表全走启发式）比：90.30 对 90.30，差 −0.003，p = 0.94。
- 和严格 fp64 比：cv350 上没有逐 token 的 fp64 结果。节点 53 全集严格 fp64 是 89.73，这里两者都高约 0.55 分，和节点 53 上
  BF16 对 fp64 的 +0.57 一致，是 BF16 推理整体的偏差，与卷积后端无关。
- 配置：SimWAM 官方 `eval_navsim.py`，`SimWAM.pt`，action-only，10 步，`seed=0`，噪声在 CPU 上用 fp32 生成，
  `cudnn.deterministic=True`、`cudnn.benchmark=False`。FlyDSL 用 SimWAM 评测里的补丁（Lumen `ccaeadc` 的
  `patch_vae_convs(min_spatial_kernel=1, video=True)`，T=1 改写和本示例相同），aiter main `78a14ae5d` 加 #7372 的 11 行 Wan2.2
  表（和 `db4c8d83` 逐字节相同），flydsl 0.5.0。navtest 按 token 切成 10 片，在 GPU 0–5 上跑。

### 2.4 查表：为什么需要 #7372 的表

FlyDSL kernel 按 `aiter/configs/model_configs/*_bf16_tuned_conv3d.csv` 查每个卷积的 launch tile。单帧编码的 30 次调用里，
25 次查表，另外 5 次是 1×1 卷积，走 matmul 快速路径，不查表。

| aiter | 查表：精确 / 借用 / 启发式 | 单帧编码（15:28–15:36 同一批，GPU 4 / GPU 5） |
|---|---|---:|
| aiter main `78a14ae5d`（没有 Wan2.2 表） | 0 / 0 / 25 | 5.686 / 5.779 ms |
| 加 #7372 `db4c8d83`（Wan2.2 表 11 行） | 25 / 0 / 0 | 5.459 / 5.660 ms |

- 没有 #7372 的表时，结果仍然正确，只是慢一些。`flydsl_conv.warn_if_untuned()` 会在日志里给出警告，列出走启发式的形状数；
  `verify_vae.py --backend flydsl` 看到启发式查表会判 FAIL，`test_flydsl_encode_matches_stock` 也会失败并说明原因。
- 这张表的 11 行正是单帧编码查到的 11 个 key：编码器的 3 个 3×3 stride 2 的 Conv2d 下采样，加 8 个 T=1 形式的卷积。
- 上表的两组耗时是 2026-11-17 下午用 SimWAM 评测里的同一个 FlyDSL 补丁测的（`ccaeadc` 的 `lumen_vae_conv.py`，T=1 改写
  和本示例相同），表当时是 59 行；删到 11 行后，单帧编码的 25 次查表逐次返回相同的配置，latent 逐位相同。那一批没有记录 host
  负载，FlyDSL 两卡相差 1.6%–3.7%，所以只用来比较同一批里的有表和没表；绝对耗时以 §2.1 为准。


### 2.5 GRPO 训练整步里的 VAE（2 卡、20 步的粗略估计）

这一节来自另一组测量，配置和上面不同：cv350 的 2 张 MI350X（GPU 4、5），Lumen-RL `d0ada15` 的
`examples/Embodied_SimWAM_NAVSIM`、SimWAM `1433ee7`、Lumen `293629f`，每卡 `batch_size=1`、`group_size=8`（官方是 4 卡），
每轮只跑 20 个 optimizer step，丢掉第 1 个 rollout 周期，统计后 16 步，也就是 4 个 rollout 周期（每个周期 1 次 rollout、
4 次 inner update，VAE 只在 rollout 里编码 1 次单帧）。2026-11-17 测，**是少量步数的粗略估计**。

| 轮 | aiter / flydsl | Lumen attention / linear | VAE 卷积 | 每个 4 步周期（中位数） | VAE 编码（rank0 / rank1） | VAE 占周期 |
|---|---|---|---|---:|---:|---:|
| gE（贴近 example） | 镜像自带 `891788643` / 0.3.2 | 开 | torch BF16 | 5.27 s（约 1.32 s/步） | 20.8 / 25.3 ms | 0.39% / 0.48% |
| gA2 | main `78a14ae5d` + #7372 `147d1a4` / 0.5.0 | 关 | torch BF16 | 2.20 s | 8.5 / 8.1 ms | 0.39% / 0.37% |
| gB2 | 同 gA2 | 关 | FlyDSL（72 个卷积全换） | 2.24 s | 6.7 / 6.8 ms | 0.31% / 0.31% |

- **VAE 只占 GRPO 整步约 0.4%**：贴近 example 的 gE 是 0.39%（rank0）和 0.48%（rank1）。
- **换 FlyDSL 的实测节省**（gA2 对 gB2，同一个 aiter main 环境，只换 VAE 卷积）：VAE 编码从 8.3 ms 降到 6.75 ms（两卡中位数），
  每次 rollout 省约 1.5 ms，摊到每个 optimizer step 约 0.4 ms，约占整步的 0.07%。两轮的整周期是 2.20 s 和 2.24 s，
  FlyDSL 那轮反而慢 1.7%，但方向和 VAE 的变化相反、量级远大于 1.5 ms，属于运行间波动，不能算 FlyDSL 变慢。
- 推断（没有实测）：把每次 rollout 省下的 1.5 ms 换到 gE 那种 5.27 s 的周期上，约省 0.03%。gE 里 VAE 编码要 20.8 ms，
  FlyDSL 在那个环境里能省多少没测，所以这个数只作量级参考。
- gE 的其余各项（每个 4 步周期）：inner update 前向 2.18 s（其中 ref 前向 1.50 s）、rollout 去噪 1.45 s、反向 0.86 s、
  old_logp 0.44 s、PDM 打分 0.13 s、video prefill 60 ms。
- FlyDSL 的 A/B 只能在 Lumen attention 和 linear 都关掉的环境里做，原因见 §3.1 的版本说明。gB2 的 FlyDSL 用的是 Lumen
  `ccaeadc` 的 `patch_vae_convs(min_spatial_kernel=1, video=True)`，T=1 改写和本示例的 `flydsl_conv.py` 相同。

### 2.6 FlyDSL 是怎么接进来的

- **Lumen 库里没有 FlyDSL 选项**：`LumenConfig(aiter_conv=True)` / `lumen.ops.conv` 走的是 aiter 的 Triton 卷积。所以本示例
  自带 `flydsl_conv.py`，做法照 Lumen main 上 `examples/perf-veomni/veomni_patches/aiter_conv.py`：直接调
  `aiter.ops.flydsl.conv_kernels.flydsl_conv_implicit`，并包一层 `_lookup_tuned_tile` 统计每次查表是精确、借用还是启发式。
  BF16、在 GPU 上、不需要梯度时走 kernel，否则回退 torch，两边都计数。
- **两处改写，都是精确的**：
  - T=1 且没有特征缓存（单帧编码，或任何编码的第一个分块）：`CausalConv3d` 在时间维前面补 `kT − 1` 个零帧，只有最后一个
    时间 tap 碰到数据，所以这次调用就等于用 `weight[:, :, -1]` 做 2-D 卷积。62 个 `CausalConv3d` 里有 60 个可以这样改写，
    另外 2 个是 stride 2 的时间下采样，不能改。
  - 其余调用保留模块自己的因果 padding 和缓存拼接，只把下面的卷积换成 kernel。
- **识别因果卷积按 `_padding` 属性（6 元组）**，不按类名。perf-veomni 的 `lumen_vae_conv.py` 按类名
  `QwenImageCausalConv3d` / `WanCausalConv3d` 匹配，认不出 SimWAM 的 `CausalConv3d`，所以不能直接拿来用。
- **Triton 路径保留**：Lumen 的惯例是库功能都通过 `LumenConfig.enable` 接入，review 也要求卷积补丁走 `lumencfg.enable()`；
  FlyDSL 目前只是示例内的补丁，不在库里。所以 `LumenConfig(aiter_conv=True)`（Triton）仍是这个示例默认的库路径，测试也保留；
  FlyDSL 作为可选后端（`verify_vae.py --backend flydsl`），本 README 的性能数据都是 FlyDSL 的。

---

## 3. Runbook

### 3.1 环境

- 镜像：`amdagi/veomni:rocm7.14_torch2.12_py3.12`（torch 2.12 + ROCm 7.14），gfx950。
- aiter：main `78a14ae5d` 或更新，加上 [ROCm/aiter#7372](https://github.com/ROCm/aiter/pull/7372) 的
  `aiter/configs/model_configs/wan22_vae_bf16_tuned_conv3d.csv`（`db4c8d83` 版，11 行）。
- flydsl 0.5.0（aiter main 的 `setup.py` 要求的版本）。
- SimWAM `1433ee7` 的 checkout，以及 `Wan2.2_VAE.pth`：

```bash
git clone https://github.com/H-EmbodVis/SimWAM.git $SIMWAM_DIR
git -C $SIMWAM_DIR checkout 1433ee7e06a62bd8b89da74455df0d82bda7b726
hf download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --local-dir $(dirname $WAN22_VAE)
export PYTHONPATH=/path/to/Lumen:$PYTHONPATH
```

`SIMWAM_DIR` 默认 `/work/SimWAM`，`WAN22_VAE` 默认 `/work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。

注意：Lumen `293629f` 和 aiter main 不兼容：CK `mha_fwd` 签名不匹配，Triton 路径缺 `get_shape_from_layout`，在 aiter main 上 Lumen 的 attention 会全部退回 torch；aiter main 的 bf16 tuned GEMM 表里也没有 GRPO 的形状，linear 同样走 torch。这不影响本示例
（VAE 的卷积补丁走本目录的 `flydsl_conv.py` 或 `lumen.ops.conv`，不用 Lumen 的 attention / linear），但在 GRPO 整步里同时开
Lumen 的 attention / linear 时要注意（§2.5 的测量用的是镜像自带的 aiter `891788643`）。

### 3.2 正确性

```bash
python3 -m pytest -v test_wan22_vae.py           # CPU 测试，加上有 GPU 时的 Triton 和 FlyDSL 测试
python3 verify_vae.py --backend flydsl --iters 50
python3 verify_vae.py --backend triton --iters 50
```

### 3.3 计时

```bash
python3 bench_encode.py --order torch,torchdet,flydsl --out round1.json   # 一张卡
python3 bench_encode.py --order flydsl,torchdet,torch --out round2.json   # 另一张卡，顺序相反
```

每轮打印三种方案的中位数、每次编码的查表结果，以及 torch.profiler 统计的 GPU kernel 时间。测之前确认卡上没有别人的进程，
并在 host 负载低的时候测：FlyDSL 的编码受 host 端限制，CPU 被占满时会慢约 10%（§2.1）。

### 3.4 验收标准

- `verify_vae.py --backend flydsl`：`flydsl_conv.patch_vae: 62 CausalConv3d (60 with the T=1 -> conv2d rewrite) and 10 Conv2d bound`；
  `FlyDSL tuned-table lookups: {'exact': 25, 'borrowed': 0, 'heuristic': 0, 'unkeyed': 0} over 16 distinct convolutions`；
  `conv calls {'aiter': 30, 'torch': 0}`；三值检查 `16 exact`；patched 对 FP32 的 SNR 不低于 stock。1e-5 那一行会是 FAIL（§2.3）。
- `pytest`：`test_flydsl_encode_matches_stock` 只在 1e-5 的 `assert_close` 上失败（调用数、16 种卷积、25 次精确查表的断言都在它之前，
  已经通过）；其余 FlyDSL 测试通过。
- 计时：两轮各自的 FlyDSL 中位数都低于同轮 torch BF16。

### 3.5 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `aiter FlyDSL conv unavailable` | aiter 太旧，或 flydsl 不是 0.5.0 | 装 aiter main `78a14ae5d` 或更新、flydsl 0.5.0 |
| 日志里有 `no exact tuned row`，或查表 `heuristic` 不为 0 | 没有 #7372 的 Wan2.2 表 | 把 #7372 的 `wan22_vae_bf16_tuned_conv3d.csv` 放进 `aiter/configs/model_configs/` |
| `conv calls` 里 `torch` 不为 0 | 输入不是 BF16、在 CPU 上，或需要梯度 | VAE 和输入都用 BF16、放在 GPU 上，在 `torch.no_grad()` 里编码 |
| FlyDSL 首次运行在 JIT 编译上停很久 | 每个新的 tile 都要编译一次 | 等它编完；磁盘缓存（`FLYDSL_RUNTIME_CACHE_DIR`）之后会复用 |
| GRPO 整步里 Lumen 的 attention / linear 退回 torch | Lumen `293629f` 和 aiter main 不兼容 | 整步训练用和 Lumen 匹配的 aiter（§2.5 用的是镜像自带的 `891788643`）；本示例的 VAE 卷积不受影响 |
| FlyDSL 编码比 §2.1 慢、变异系数超过 2% | host 的 CPU 负载高（FlyDSL 编码受 host 端限制，§2.2） | 用 `uptime` 看 load average，负载低时重测；对比时让 torch 和 FlyDSL 在同一进程里交替测 |
| flydsl 0.5.0 计算 bitcode hash 时报错找不到文件 | ROCm SDK 里有断开的 `*.bc` 软链接 | 用去掉断链的 ROCm 目录副本，设 `ROCM_PATH` 指过去 |

---

## 4. 文件索引

| 文件 | 作用 |
|---|---|
| `wan22_vae.py` | 从 SimWAM checkout 加载 `wan_video_vae.py` 和权重（strict）；GRPO shape；`ConvCalls`（Triton 路径的 aiter / torch 计数）；Triton 路径的三值检查 |
| `flydsl_conv.py` | 本示例的 FlyDSL 卷积：`patch_vae`、T=1 改写、调用和查表计数、`warn_if_untuned`、`Trace`、三值检查 |
| `verify_vae.py` | 单卡验证，`--backend triton` / `flydsl`：调用数、查表、1e-5、对 FP32 的 SNR、耗时、三值检查 |
| `bench_encode.py` | 单帧编码的同卡 A/B 计时（torch、torch deterministic、FlyDSL）和 torch.profiler 的 kernel 时间 |
| `test_wan22_vae.py` | pytest：CPU 的结构、绑定和 T=1 改写；GPU 的 Triton 与 FlyDSL 测试 |
| `README_EN.md` | 本文的英文版 |
