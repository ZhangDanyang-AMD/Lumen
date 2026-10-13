# Qwen3-30B-A3B FSDP FP8 优化过程

Megatron FP8 见 [`qwen3-30b-a3b-fp8-megatron-optimization.md`](qwen3-30b-a3b-fp8-megatron-optimization.md)。

8×MI350X 上，Qwen3-30B-A3B FSDP FP8 blockwise2d 预训练（MBS=2，GBS=256，seq=4096，steps 11–20 中位数）从 **13.21 s / 19.38 samples/s** 降到 **9.54 s / 26.85 samples/s**（−3.68 s，−27.8%，1.39×）。同一形状的 Megatron FP8 是 8.40 s，这条 FSDP 慢 **1.14 s**。Megatron BF16 是 9.88 s，这条 FSDP FP8 快 **0.34 s**。

## 1. 口径

Qwen3-30B-A3B：48 层，hidden 2048，32 个 Q head，4 个 KV head，head dim 128，128 experts，top-8，expert FFN 768。栈是 HuggingFace + FSDP2 + Lumen SonicMoE，TP=PP=CP=1，DP=8，EP=8，`shard_grad_op`。同一 8 卡上 microbatch 数是 16。SonicMoE 替换本地专家 MLP。Router 仍是 BF16 `aten::mm`。EP all-to-all 传 BF16 行。

FP8 是 OCP E4M3（`torch.float8_e4m3fn`），官方 blockwise2d：权重 128×128，激活动态 1×128。QKV、proj、专家 w1（逐列 gate/up）和 w2 的前向、dgrad、wgrad 走这条量化。master 权重保持 BF16。embed、lm_head、norm、router 保持 BF16。Attention 保持 BF16：前向 OPUS，反向 AITER `fmha_v3_bwd` atomic16。FlyDSL `sonic_moe` 只吃 BF16，这条路径的专家 GEMM 是 Triton grouped FP8。FSDP 的 `_SonicLocalExperts` 只接受 `SONIC_MOE_GEMM_BACKEND=triton`。

all-to-all 之后每个 rank 的专家 GEMM 大约是 65536 token、16 个本地专家。Attention 的 QKV / proj 是本地 8192 行（MBS×seq）。每种 GEMM 每步 48 层 × 16 microbatch = 768 次。专家 wgrad 按 expert 分段量化，scale 不跨专家边界。RCCL 不接受 `float8_e4m3fn`。

数字都是真实权重加 FineWeb 的 20 步日志，steps 11–20 中位数（偶数个样本取排序后第 5、第 6 个的平均），samples/s = 256 / 中位秒。前 3 步含加载和 autotune，不进统计。步时噪声大约 50 ms，小于这个的孤立收益不留。Profiler 打开的 step 墙钟不是速度数。

起点是 2026-10-09 的 blockwise2d，中位 13.21 s。它已经带上前一天的专家计数交换：split size 在侧流上拷回，不再用默认流上的 `tolist()` 把 gather 抽干。表里每一行是当时那次 20 步，叠在上一行的代码上。

step 1 的 lm loss 锁在 **2.562195**。grad norm 在起点和 transpose 那一档是 6.579，expert tile 那一档是 6.566，从 Gluon 起是 **6.563**，最后一档 aux histc 是 **6.564**。step 20 lm loss 落在 2.4122–2.4132。最新一档是 2.412354、grad norm 0.553。这和 Megatron FP8 的 2.561880 / 6.515、step 20 的 2.4128 是两条数值路径，只比墙钟。

## 2. 优化过程

每一行叠在上一行上，形状始终是 MBS=2、GBS=256。

| 阶段 | 做了什么 | 中位 (s) | samples/s | 相对上一步 | step 20 lm loss |
|---|---|---|---|---|---|
| **起点** | 计数交换已重叠的 blockwise2d。步时分成大约 12.3 s 和 14–16 s 两簇 | 13.21 | 19.38 | — | 2.412720 |
| **FP8 transpose** | blockwise 反向的 FP8 转置换成 tiled kernel | 12.08 | 21.19 | −1.13 s | 2.412292 |
| **expert tile** | 专家 GEMM 对齐 128 的 scale block，加宽 wgrad tile | 11.34 | 22.57 | −0.74 s | 2.412415 |
| **Gluon QKV** | QKV 和 proj 的 blockscale GEMM 换 Gluon | 10.98 | 23.31 | −0.36 s | 2.413208 |
| **手动 GC** | 启动时回收一次，然后关掉自动回收 | 10.95 | 23.37 | −0.03 s | 2.412964 |
| **交错 SwiGLU** | 按 FSDP 的逐列 gate/up 一次加载一对 | 10.61 | 24.12 | −0.34 s | 2.412903 |
| **gather tile** | MoE gather / reduce 改成一行一个 program | 10.47 | 24.45 | −0.14 s | 2.412903 |
| **融合 QKV** | Q、K、V 合成一次 GEMM，激活只量化一次 | 10.34 | 24.75 | −0.13 s | 2.412659 |
| **apex RoPE** | 未融合的 cos/sin 换成 apex 原生 fused RoPE | 10.03 | 25.53 | −0.32 s | 2.412231 |
| **dense 权重量化一次** | QKV 和 proj 的 FP8 权重跨 16 个 microbatch 复用 | 9.83 | 26.04 | −0.20 s | 2.412842 |
| **双操作数分段量化** | 专家 wgrad 的两份激活一次量化 | 9.79 | 26.15 | −0.04 s | 2.412964 |
| **延迟 loss `.item()`** | microbatch 的 loss 留在设备上，记日志时再同步 | 9.76 | 26.22 | −0.03 s | 2.412476 |
| **dispatch histc** | 专家计数不再用会同步的 `bincount` | 9.67 | 26.48 | −0.10 s | 2.412415 |
| **aux histc** | auxiliary loss 的专家直方图同样留在设备上 | **9.54** | **26.85** | −0.13 s | 2.412354 |

Gluon 之前，steps 11–20 分成两簇。起点的快簇大约 12.3 s，慢簇 14.6–16.4 s。Gluon 的快簇是 10.95–10.98 s，慢簇大约 14.3 s，四次慢、六次快，中位数落在快簇上。手动 GC 之后十个样本都在 10.929–10.970 s。后面每一档的十个样本都落在大约 50 ms 以内。

**FP8 transpose（13.21 → 12.08 s）。** blockwise 反向要把 FP8 操作数转成连续布局，`aten` 的 float8 拷贝整段露在步时上。tiled transpose 和它逐位一致。慢簇还在。

**expert tile（12.08 → 11.34 s）。** 原来的 FSDP 配置是 `BLOCK_K=64`，wgrad tile 也偏窄。改成和 128 scale block 对齐的配置。这一档的 step 1 是 lm loss 2.562134、grad norm 6.566。下一档 Gluon 回到 2.562195 / 6.563，之后不再动。四个无 gather 的前向 / dgrad 形状是 `BLOCK_M=BLOCK_N=BLOCK_K=128`，4 warps，2 stages。wgrad 是 `BLOCK_K=BLOCK_N=BLOCK_T=128`，8 warps。更大的 tile、`BLOCK_K=64` 和 3 stages 都更慢，没有再改。

**Gluon QKV（11.34 → 10.98 s）。** QKV 和 proj 的 blockscale GEMM 在 gfx950 上走 Gluon。Triton blockscale 在这几个形状上更慢。中位数少了 0.36 s，但慢簇还在，均值仍是 12.29 s。

**手动 GC（10.98 → 10.95 s）。** 中位数几乎不动。去掉的是慢簇：Gluon 有四步大约 14.3 s，GC 之后最长的一步是 10.970 s。自动回收落进步时，一次大约拉长 3.4 s。启动时回收一次，然后把自动回收关掉。

**交错 SwiGLU（10.95 → 10.61 s）。** FSDP 的 `w1` 是逐列的 gate/up，不是 Megatron 的 `[gate | up]`。按列对加载一次，比通用 GLU kernel 少一次扫描。`concat_layout=False` 必须和这个布局一致，反了会静默算成 `silu(up) * gate`。

**gather tile（10.61 → 10.47 s）。** 256 宽的 tile 在 2048 宽的 hidden 行上要起 8 个 program。改成一行一个 program。

**融合 QKV（10.47 → 10.34 s）。** Q、K、V 原来各量化一次同一份激活。三个投影合成一次 blockwise GEMM，再按宽度切开。

**apex RoPE（10.34 → 10.03 s）。** 未融合的 cos/sin 落在步时上。apex 原生 fused RoPE 和这条布局的舍入一致。AITER RoPE 在 import 时按 `USE_ROCM_AITER_ROPE_BACKEND` 选定，那次 20 步中位 11.02 s，而且十个样本从 10.52 s 散到 12.73 s，没有留下。开关是 `LUMEN_APEX_ROPE=1`，`USE_ROCM_AITER_ROPE_BACKEND=0`。

**dense 权重量化一次（10.03 → 9.83 s）。** QKV 和 proj 的 BF16 权重在 16 个 microbatch 里不变。每个 microbatch 都做一次 FP8 量化和转置。缓存 FP8 权重和转置，`optimizer.step` 之后丢掉。开关 `LUMEN_DENSE_WEIGHT_QUANT_ONCE=1`。专家权重是 FSDP 的 all-gather view，不是 `nn.Parameter`，这条缓存碰不到它们。

**双操作数分段量化（9.83 → 9.79 s）。** 专家 wgrad 的两份操作数原来各做一次 segment 量化，scale 不能跨专家。合成一次读取。先有一版中位 9.82 s，少 8 ms，落在噪声里；留下的是 9.79 s 这一版。FP8 值和分开量化一致。

**延迟 loss `.item()`（9.79 → 9.76 s）。** 每个 microbatch 三次 `.item()` 把 GPU 队列抽干。loss 留在设备上，记日志时再做一次同步。中位数少 29 ms。

**dispatch histc（9.76 → 9.67 s）。** `torch.bincount` 即使给了 `minlength` 也会把 min/max 读回 host，all-to-all 之前把流抽干。专家 id 落在 `[0, 128)` 里，`histc` 写出同样的桶，gather 可以接着排队。

**aux histc（9.67 → 9.54 s）。** HuggingFace 的 auxiliary loss 在 `attention_mask is None` 时对每一层做 `bincount`。48 层 × 16 microbatch = 768 次，profile 里这组调用的 CPU 时间是 169 ms。换成 `histc` 之后 loss 和梯度逐位一致。带 mask 的路径仍是原来的 `scatter_add_`。开关 `LUMEN_ASYNC_AUX_COUNTS=1`。

当前最快日志：`examples/qwen3-30b-a3b/results/qwen3-30b-a3b-fsdp-sonic-async-aux-counts-fp8_blockwise2d-seq4096-mbs2-gbs256.log`（中位 9.536 s，steps 11–20 均值 9.536 s，十个样本 9.511–9.560 s）。step 1 是 lm loss 2.562195、grad norm 6.564，step 20 是 2.412354、0.553。峰值显存 143674 MB。

## 3. 没有留下的

| 尝试 | 结果 |
|---|---|
| AITER fused RoPE | 中位 11.02 s，比当时的 apex 前一档 10.34 s 更慢，十个样本从 10.52 s 散到 12.73 s |
| score 反向融合 | 中位 10.98 s，和当时的 Gluon 10.98 s 一样，慢簇还在 |
| 专家梯度累加 | 中位 10.35 s，相对当时的融合 QKV 10.34 s 没有少 |
| all-to-all 改传 FP8 | 1 步对照的 step 1 是 lm loss 2.640747、grad 6.500。整段 FP8 载荷变成 2.648315 / 5.155。只改反向是 2.640747 / 5.013 |
| 专家 FP8 权重按 step 缓存，dgrad 用连续转置 | 孤立测量大约少 74 ms 量化和 38 ms dgrad，输出逐位一致。3 步的 step 2/3 是 9.570 / 9.553 s，对照是 9.548 / 9.540 s。step 3 lm loss 从 2.611450 变成 2.604126，grad 从 6.316 变成 5.981。峰值多大约 3.5 GB |
| Router GEMM 换 hipBLASLt | 三条 router GEMM 合计 36 ms/step，换成 hipBLASLt 是 37 ms |
| FlyDSL `sonic_moe` | 只有 BF16。接进来会把专家计算从 blockwise FP8 换成 BF16，step 1 不再是 2.562195 |
| CK deepgemm | 这份测试在非 gfx942 上直接返回，scale 也不是 1×128 / 128×128 |
| SwiGLU 前向再融合行量化 | 激活的 BF16 还要留给反向，融合只少 0.84 µs |
| 专家 tile 再扫 | 128×128、4 warps、`BLOCK_K=128` 在四个无 gather 形状上最快。wgrad 8 warps、`BLOCK_K=128` 最快 |

all-to-all 的协议、channel、MSCCL、关 IB、设备 IPC、pad 到 capacity、跨 microbatch 重叠，都试过，要么 loss/grad 对不上，要么显存或中位数没有收益。这条没有重开。

## 4. 还差在哪里

9.54 s 再往下，现有 kernel 的 tile 和 host 同步都已经扫过。profile 取的是 aux histc 之前的一步（当时稳态大约 9.67 s）。Self CUDA 合计 9.48 s，和墙钟是同一量级，所以这段设备时间基本是露在步时上的。aux histc 少掉的 0.13 s 是 host 同步，不在下面这些 kernel 里。

还露在外面、而且现有 kernel 缩不动的，主要是：

- all-to-all，3068 次，独占 **2.69 s**。Megatron 那条 1F1B 把大约 5.3 s 的 all-to-all 盖在计算下面，露出来的空闲大约 0.3 s。FSDP 这条没有那层重叠，2.69 s 在关键路径上。载荷改成 FP8 会改 step 1 的 loss 和 grad，没有留下。
- 专家 grouped GEMM。前向和 dgrad 的 `_grouped_gemm_kernel` 是 0.887 s（3068 次，289 µs）。wgrad 的 `_grouped_gemm_dw_kernel` 是 0.867 s（1532 次，566 µs）。tile 已经扫完。换实现要同时吃 1×128 激活 scale 和 128×128 权重 scale，并且超过这两个时间。FlyDSL 没有这套 FP8 kernel。
- dense A8W8 `_gemm_a8w8_blockscale_kernel`，**0.72 s**。QKV 已经融合，Gluon 和按 step 缓存权重都在。
- Attention 反向 `fmha_bwd_hd128_bf16_causal_br_a16_psskddv`，**0.55 s**。atomic16 已经是这条反向里更快的配置。
- MoE sort。`_sort_chunks_by_map_kernel` 0.375 s，chunk-sort 反向 0.201 s。拷贝靠近带宽地板，只改行映射进不了中位数。
- FP8 量化。行量化 0.242 s，专家 wgrad 的双操作数分段量化 0.231 s。SwiGLU 再融合行量化省不下，因为 BF16 激活还要留给反向。

lm_head 的三次 BF16 GEMM 大约 0.24 s，在 loss 路径上，改成 FP8 会动数值。Router 的三条 GEMM 合计 36 ms。`aten::add_` 0.29 s 和一批 elementwise 大约 0.23 s，没有单独一颗 kernel 值得再跑 20 步。
