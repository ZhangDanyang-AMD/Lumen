# Qwen3-30B-A3B Megatron FP8 优化过程

BF16 combined 1F1B 见 [`qwen3-30b-a3b-bf16-megatron-optimization.md`](qwen3-30b-a3b-bf16-megatron-optimization.md)。

8×MI350X 上，Qwen3-30B-A3B Megatron FP8 blockwise2d 预训练（MBS=2，GBS=256，seq=4096，steps 11–20 中位数）从 **12.70 s / 20.16 samples/s** 降到 **8.40 s / 30.46 samples/s**（−4.30 s，−33.8%，1.51×）。同一形状的 Megatron BF16 是 9.88 s，这条 FP8 快 **1.48 s**。

## 1. 口径

Qwen3-30B-A3B：48 层，hidden 2048，32 个 Q head，4 个 KV head，head dim 128，128 experts，top-8，expert FFN 768。栈是 Megatron-LM + Lumen + SonicMoE，TP=PP=CP=1，DP=8，EP=8。同一 8 卡上 `num_microbatches = 16`。SonicMoE 只替换 `MoELayer.experts`，router、permute 和 EP all-to-all 仍是 Megatron。

FP8 是 OCP E4M3（`torch.float8_e4m3fn`），官方 blockwise2d：权重 128×128，激活动态 1×128。QKV、proj、专家 w1（gate+up）和 w2 的前向与反向走这条量化。master 权重保持 BF16。embed、lm_head、norm、router 保持 BF16。Attention 保持 BF16：前向 OPUS，反向 AITER `fmha_v3_bwd` atomic16。FlyDSL 只吃 BF16，这条路径的专家 GEMM 是 Triton grouped FP8。

all-to-all 之后每个 rank 的专家 GEMM 大约是 65536 token、16 个本地专家。Attention 的 QKV / proj 是本地 8192 行（MBS×seq）。每种 GEMM 每步 48 层 × 16 microbatch = 768 次。

数字都是真实权重加 FineWeb 的 20 步日志，steps 11–20 中位数（偶数个样本取排序后第 5、第 6 个的平均），samples/s = 256 / 中位秒。前 3 步含加载和 autotune，不进统计。步时噪声大约 50–100 ms，小于这个的孤立收益不留。Profiler 打开的 step 墙钟不是速度数。

起点是 2026-10-08 在已经叠完 BF16 栈（CK/OPUS attention、attention CUDA graph、EP overlap、`CONN=8`、异步 split、Q RMSNorm、router 反向 bf16、dQ atomic16）上重测的 blockwise2d，中位 12.70 s。BF16 那份记录里的 12.95 s 是 9 月 28 日的第一次测量，step 11–20 里还有一次 66 s 的停顿，不作为本系列的起点。

step 1 在 wgrad tile 之后从 lm loss 2.562199、grad norm 6.528 收到 2.561856 / 6.515，随后锁在 **2.561880 / 6.515**。专家置换之后是 **2.561891 / 6.516**，相对 BF16 ulp（约 2.5 处是 0.015625）可以忽略。step 20 lm loss 落在 2.4127–2.4128。BF16 FlyDSL 的 step 20 是 2.4103，两边是两条数值路径，只比墙钟。

## 2. 优化过程

每一行叠在上一行上，形状始终是 MBS=2、GBS=256。

| 阶段 | 做了什么 | 中位 (s) | samples/s | 相对上一步 | step 20 lm loss |
|---|---|---|---|---|---|
| **起点** | BF16 栈 + blockwise2d。专家 wgrad 仍是会溢出的 BF16 tile | 12.70 | 20.16 | — | 2.412755 |
| **wgrad tile** | 专家 wgrad 的 tile 对齐 128 的 scale block | 11.93 | 21.46 | −0.77 s | 2.412776 |
| **fwd/dgrad K=128** | 前向和 dgrad 每个 K 迭代走一个 scale block | 11.73 | 21.83 | −0.20 s | 2.412713 |
| **融合 grad 量化** | 同一份专家 grad 只量化一次，给 dgrad 和 wgrad 共用 | 11.57 | 22.12 | −0.16 s | 2.412809 |
| **Gluon QKV** | gfx950 上 QKV 和 proj 的 blockscale GEMM 换 Gluon | 11.40 | 22.45 | −0.17 s | 2.412803 |
| **FP8 transpose** | Gluon 要的连续 FP8 操作数改走 tiled transpose | 11.05 | 23.17 | −0.35 s | 2.412670 |
| **score 反向** | 专家输出缩放的 token / score 梯度合成一个 kernel | 10.54 | 24.29 | −0.51 s | 2.412802 |
| **apex RoPE** | 未融合的 cos/sin 换成 apex 原生 fused RoPE | 10.07 | 25.41 | −0.47 s | 2.412846 |
| **fp32 wgrad** | 专家 wgrad 直接累加进 fp32 `main_grad` | 9.97 | 25.67 | −0.10 s | 2.412821 |
| **手动 GC** | 稳态步之前做一次垃圾回收，之后关掉自动回收 | 9.54 | 26.83 | −0.43 s | 2.412805 |
| **专家置换** | 按层把专家换到各 EP rank，拉平 all-to-all | 8.53 | 30.02 | −1.01 s | 2.412825 |
| **w1 N=128** | w1 wgrad 的 N tile 从 64 加到 128 | **8.40** | **30.46** | −0.13 s | 2.412845 |

手动 GC 之前，steps 11–20 分成两簇，大约差 0.7 s。中位数落在两簇中间。GC 收掉之后簇合成一条，8.53 s 和 8.40 s 两档各自的 10 个样本都落在大约 30 ms 以内。

**wgrad tile（12.70 → 11.93 s）。** blockwise 的 scale 是每 128 个 token 一行。原来的 BF16 tile 把 FP8 累加器挤出寄存器。wgrad 的 `BLOCK_T` 和 `BLOCK_K` 收到 128，和 scale block 对齐。step 1 lm loss 从 2.562199 变到后面几档的 2.561856 附近，是累加路径变了，不是样本变了。

**fwd/dgrad K=128（11.93 → 11.73 s）。** 四个无 gather 的专家前向 / dgrad 配置改成 `BLOCK_K=128`，`BLOCK_M=BLOCK_N=128`，4 warps，2 stages。每个 K 迭代吃一个 scale block。更大的 M/N tile 和 8 warps 在后来的孤立扫描里都更慢，没有再改。

**融合 grad 量化（11.73 → 11.57 s）。** dgrad 和 wgrad 原先对同一份激活梯度各做一次 segment 量化。合成一次读取，两份 scale 一起写出。

**Gluon QKV（11.57 → 11.40 s）。** QKV 是 `[8192, 5120] × [5120, 2048]` 这一档，proj 是 `[8192, 2048] × [2048, 4096]`。gfx950 上 `gemm_a8w8_blockscale` 走 Gluon，默认 M>2048 用 128×256。Triton blockscale 在这几个形状上更慢。后来按 tile 再扫过一遍，默认已经是最快的。

**FP8 transpose（11.40 → 11.05 s）。** Gluon 不能加载非单位 stride 的 FP8，转置必须物化成连续张量。`aten` 转置换成 `fast_transpose_2d`。这些拷贝还在，只是单次更短。全部删掉的暴露上限后来测到大约 68 ms，进不了中位数。

**score 反向（11.05 → 10.54 s）。** 专家输出乘路由分数的反向，原先是两次 hidden 大小的乘加再加一次归约。合成一个 kernel。

**apex RoPE（10.54 → 10.07 s）。** 未融合的 cos/sin 落在步时上。apex 原生 fused RoPE 和它逐位一致。AITER RoPE 在 import 时按 `USE_ROCM_AITER_ROPE_BACKEND` 选定，舍入差到 0.03125，没有留下。开关是 `LUMEN_FUSED_ROPE=1`，脚本在 `torchrun` 之前把该环境变量设成 0。Qwen 的 `rotary_interleaved` 是关的。

**fp32 wgrad（10.07 → 9.97 s）。** grouped wgrad 的 epilogue 原来把 bf16 舍入后的 tile 写进 `main_grad`。改成 fp32 累加，一个 writer，不用 atomic。返回给 DDP 的是一块占位 bf16，hook 看到梯度已经进了 `main_grad` 就跳过第二次累加。开关 `LUMEN_EXPERT_WGRAD_ACCUM=1`。

**手动 GC（9.97 → 9.54 s）。** 拆步之后看到慢步里有 720–800 ms 的垃圾回收，快步只有 26–35 ms，优化器本身大约 22–24 ms。稳态开始前回收一次，然后把自动回收间隔设成 0。两簇之间大约 0.75 s 的空隙消失。开关在 `run_qwen3_30b_a3b_megatron.sh` 里默认打开。

**专家置换（9.54 → 8.53 s）。** 热门专家堆在同一个 EP rank 上时，那一次 all-to-all 落在关键路径上。按 FSDP 那条已经验证过的表，把 48×128 的专家换到各 rank，rank `r` 拿槽 `[16r, 16r+16)`。Router 在 aux loss 和 expert bias 更新之后再用逆表 `index_select`，aux loss 仍按逻辑专家号算。权重是就地 `copy_`，随后 `optimizer.reload_model_params()`，让 fp32 master 跟上。token 仍进同一个专家，combine 的归约顺序变了，所以 step 1 lm loss 从 2.561880 变到 2.561891，grad norm 仍是 6.516。置换前 profile 里 dispatch 反向有一段 582 ms 的 all-to-all 尾巴，置换后最大槽位差大约 67 ms。开关 `LUMEN_EXPERT_REMAP=1`。这套 20 步配方是 `--finetune --no-load-optim`，不存 checkpoint；存盘会把已经置换的局部布局再写出去。

**w1 N=128（8.53 → 8.40 s）。** w1 wgrad 的输出 tile 在 N 方向每块都要把激活行再读一遍。`BLOCK_N` 从 64 收到 128，重读次数从 24 次降到 12 次。孤立测量 0.695 ms → 0.540 ms，768 次大约 120 ms，墙钟少了 125 ms。w2 原来就是 `BLOCK_N=128`，扫描里已经最快，没有改。当前配置是 `(N, K, E) = (1536, 2048, 16)`，`BLOCK_K=128`，`BLOCK_N=128`，`BLOCK_T=128`，8 warps，2 stages。

当前最快日志：`examples/qwen3-30b-a3b/results/qwen3-30b-a3b-sonic-fp8-blockwise2d-dw-n128-seq4096-mbs2-gbs256.log`（中位 8.404 s，steps 11–20 均值 8.403 s）。step 1 是 lm loss 2.561891、grad norm 6.516，step 20 是 2.412845、0.566，mem 0.7983。

## 3. 没有留下的

| 尝试 | 结果 |
|---|---|
| 更宽的 permute tile | 11.32 s，相对当时的 11.40 s 少 83 ms，落在噪声里 |
| Dispatch 改传 FP8 行 | 12.70 s，比当时的 11.40 s 多 1.30 s |
| AITER fused RoPE | 绝对误差 0.03125，比 apex 原生慢，舍入路径不同 |
| QKV / proj 再扫 Gluon tile | 默认 128×256 最快。`BLOCK_M=256` 编译失败 |
| lm_head 换后端或 split-K | 三次 BF16 GEMM 各约 4.4–4.8 ms，CK / `torch.mm` 已经最快。16 个 microbatch 合计约 0.22 s，几乎全暴露 |
| 专家 fwd / dgrad 再扫 tile | 128×128、4 warps、`BLOCK_K=128` 在四个形状上都最快 |
| MoE chunk-sort 的行映射 | 映射自己 65 µs → 11 µs，20 步中位 8.386 s，只少 18 ms。拷贝本身大约 113 µs，靠近带宽地板 |
| 去掉全部 FP8 transpose | 暴露大约 68 ms |
| 量化 kernel 再合成到零耗时 | 量化这一族暴露大约 193 ms，融合消不掉读取和 scale |
| 去掉专家权重那几次 `copy_` | w1+w2 暴露 3.3 ms，已经被别的计算盖住 |
| Gradient-accumulation fusion | BF16 上已经更慢（10.71 s 对 10.66 s），FP8 没有打开 |
| 残差加进下一层 RMSNorm、K 也换 Triton、专家 GEMM 切块、`--delay-wgrad-compute`、IPC / MORI / channel 扫描、pad 到 capacity | BF16 上已经否定，这条没有重开 |

手动 GC 之前的拆步日志（`qwen3-30b-a3b-sonic-fp8-blockwise2d-step-split-seq4096-mbs2-gbs256.log`，中位 9.95 s）是诊断，不是一档速度。它只用来确认慢步里的 0.7 s 是垃圾回收。

## 4. 还差在哪里

8.40 s 再往下，现有 kernel 的 tile、量化和拷贝都已经扫过。置换之后的 profile（step 12，开了 profiler，墙钟 38.2 s，不是速度数）里，GPU 上 kernel 区间的并集是 8.30 s，和当时 8.53 s 的稳态墙钟一致。下面的独占时间来自这次 profile，wgrad 那段已经扣掉后来 N=128 落到墙钟上的 125 ms。

还露在外面、而且现有 kernel 缩不动的，主要是两段：

- Attention 反向 `fmha_bwd_hd128_bf16_causal_br_a16_psskddv`，768 次，独占大约 **0.49 s**。BF16 报告里同类 kernel 大约是 2.3 PFLOP/s 的 30%。打满峰值大约拿回 0.35 s，步时到 **8.06 s**。现实的 2 倍大约拿回 0.25 s，步时到 **8.16 s**。gfx950 上这条反向只有一个编译好的 kernel，没有第二套 tile。
- w1 wgrad 现在 0.540 ms/次。这份输出把激活行按 N 块重读，流量大约 1.86 GB，8 TB/s 的地板大约 0.23 ms。768 次的差额大约 **0.24 s**，步时到 **8.17 s**。`BLOCK_N=256` 在孤立扫描里更慢。w2 已经是扫描里最快的配置。

两段同时靠近各自的屋顶，大约是 **7.8 s**。只做其中一个，步时仍在 8.1 s 以上。all-to-all 在置换之后还暴露大约 0.26 s，tile 缩不了它。Host 发射和 GPU 并集之间大约还有 0.22 s，N=128 那 125 ms 是 1:1 进了墙钟的，所以这段空隙不是把 kernel 收益藏起来的原因。

lm_head 三次 BF16 GEMM 独占大约 0.23 s，vocab 151936，没有更快的后端。SwiGLU 独占大约 22 ms。MoE sort 的拷贝靠近带宽地板，只加速行映射已经在 20 步里证明进不了中位数。
