# Qwen-Image FlowGRPO 全参数 RL 训练：基础版与 Lumen 加速版

> English version: [README_EN.md](README_EN.md)

在单节点 8× AMD Instinct MI350X 上复现 verl-omni 官方配方
[`run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh)：
Qwen-Image（20B DiT）全参数 FlowGRPO，VeOmni 训练引擎，vLLM-Omni rollout，Qwen3-VL-8B 作 OCR reward，300 步。

- **基础版**：官方配方，只改路径、卡数和 ROCm 注意力后端（§4）。
- **Lumen 版**：在基础版之上，给 actor（VeOmni 训练引擎）加 3 项 Lumen 补丁（§5），其余完全相同。

## 1. 结果

8× MI350X，同一 seed。验证 reward（OCR 测试集 1018 条）：

| step | 官方参考 | 基础版 | Lumen 版 |
|---:|---:|---:|---:|
| 0 | 0.406 | 0.402 | 0.403 |
| 30 | | 0.644 | 0.619 |
| 60 | | 0.815 | 0.864 |
| 90 | | 0.901 | 0.940 |
| 120 | 0.937 | 0.928 | |
| 300 | | 0.949 | |

性能（第 2–90 步中位数，不含验证和存档步）：

| | 基础版 | Lumen 版 |
|---|---:|---:|
| actor（`old_log_prob` + `update_actor`） | 31.1 s | 27.7 s（−11%） |
| 整步 | 129.5 s | 125.2 s（−3.3%） |
| 吞吐（images/GPU/s） | 0.494 | 0.511 |

- 官方参考取自 verl-omni [`docs/algo/performance.md`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/docs/algo/performance.md)：
  验证曲线来自 VeOmni 引擎 64× H100，吞吐 0.510 images/GPU/s 来自 4× H200。
- 基础版 300 步全程（含验证与存档）约 12.5 h；Lumen 版跑了 90 步。
- Lumen 只加速 actor，actor 约占每步的 24%；生成（约 53 s）和 OCR 打分（约 39 s）不受影响，所以整步收益约 3%。
- 学习曲线：Lumen 版与基础版走势一致（第 61–90 步训练 reward 0.776 对 0.770）。同一配置的不同运行之间本身
  就有明显波动，比较时请用同一台机器、同一 seed。

## 2. 硬件与环境要求

- 8× MI350X（gfx950，288 GB），单卡显存峰值约 208 GB。
- 主机内存 ≥ 1.5 TB：8 个 rollout 副本休眠时整条 pipeline 留在主存（约 900 GB）。
- `$WORK` 所在磁盘 ≥ 1 TB：模型 71 GB；每个 checkpoint 228 GB，保留最近 2 个。
- Docker；宿主机 ROCm 驱动需兼容镜像内的 ROCm 7.2；能访问 GitHub、Hugging Face、Docker Hub、PyPI。

## 3. 复现步骤

```bash
export WORK=/path/to/work                       # 代码、模型、数据、运行结果都放在这里
git clone https://github.com/ZhangDanyang-AMD/Lumen.git $WORK/src/Lumen
EX=$WORK/src/Lumen/examples/perf-veomni/rl-flowgrpo-qwen-image

docker pull vllm/vllm-openai-rocm:v0.27.0
WORK=$WORK bash $EX/launch_container.sh         # 容器名 qwen-image-rl
docker exec qwen-image-rl bash $EX/setup_env.sh # 约 30 分钟；末尾应输出 19653 / 1018 行与 SETUP OK
```

训练（两个版本各用 8 卡，依次运行；日志在 `$WORK/runs/<RUN_NAME>/train.log`）：

```bash
docker exec -d qwen-image-rl bash -c "RUN_NAME=base bash $EX/run_flowgrpo.sh"           # 基础版
docker exec -d qwen-image-rl bash -c "RUN_NAME=lumen LUMEN=1 bash $EX/run_flowgrpo.sh"  # Lumen 版
docker exec qwen-image-rl python3 $EX/compare.py $WORK/runs/base $WORK/runs/lumen      # 与官方、彼此对比
```

- 快速对比：在命令末尾加 `trainer.total_training_steps=90`，第 30/60/90 步验证，每个版本约 3.5 h（上表 Lumen 版即此设置）。
- W&B：给 `docker exec` 加 `-e WANDB_API_KEY=<key> -e WANDB_PROJECT=<project>` 即在线上传；不设则离线记录在运行目录里。
- 确认 Lumen 生效：`docker exec qwen-image-rl bash -c 'grep -h lumen-rl /tmp/ray/session_latest/logs/worker-*.out | head'`
  应能看到 `installed ... ['mask_all_true_to_none', 'rmsnorm_fuse', 'local_adamw']` 和非零的调用计数。
- 停止运行：`docker exec qwen-image-rl bash -c 'ray stop --force; pkill -9 -f VLLM'`，再用 `rocm-smi --showpids` 确认 GPU 已空。

## 4. 与官方配方的差异

算法与超参数不改动。只有以下几项不同：

| 项 | 官方 | 本示例 | 原因 |
|---|---|---|---|
| GPU 数 | 脚本默认 4 | 8（`NUM_GPUS`） | 单节点 8 卡 |
| actor 参数 / 优化器 offload | 开 | 关 | 288 GB 显存放得下；开启后主存不够，且 offload 拷贝很慢 |
| 注意力后端 | 默认 | actor `native`（torch SDPA），rollout `TORCH_SDPA`、`enforce_eager=True` | ROCm 兼容 |
| 训练前验证 | 关 | 开 | 得到第 0 步的对比点 |
| seed | 默认 | 数据与 rollout 固定为 42 | 便于同 seed 对比 |

所有版本均已固定：镜像 `vllm/vllm-openai-rocm:v0.27.0`，verl、verl-omni、VeOmni、vllm-omni、flow_grpo 的 commit，以及两个模型的 revision，都写在 `setup_env.sh` 里。

## 5. Lumen 补丁（`LUMEN=1`）

补丁只在构建 VeOmni 训练引擎的 actor 进程里安装；rollout 和 judge 在各自的进程里，不受影响。

| `LUMEN_PATCH` | 作用 | 数值 |
|---|---|---|
| `mask_all_true_to_none` | 训练 micro-batch 内是同一 prompt 的样本，文本 mask 全为 True；去掉这个 mask，让注意力走 flash kernel | 数学上等价；flash 路径是确定性的 |
| `rmsnorm_fuse` | diffusers RMSNorm 前向经 `torch.compile` 融合（[`../veomni_patches/rmsnorm.py`](../veomni_patches/rmsnorm.py)） | 相对 FP32 为 55.6 dB，原实现为 52.6 dB |
| `local_adamw` | fused AdamW 直接在 FSDP2 本地分片上执行（[`../veomni_patches/local_adamw.py`](../veomni_patches/local_adamw.py)） | 与原实现逐位一致 |

perf-veomni 的其余优化不用于 RL：actor 不运行 VAE，所以 VAE 相关优化用不上；关梯度检查点和关 reshard
会占用 rollout 所需的显存；`sdpa_efficient` / `attn_triton_fwd` 所用的 aiter 融合反向，在 RL 的真实激活上会产生非有限梯度。

## 6. 文件

| 文件 | 运行位置 | 作用 |
|---|---|---|
| `launch_container.sh` | 宿主机 | 启动容器，把 `$WORK` 挂载到相同路径 |
| `setup_env.sh` | 容器内 | 拉取固定版本的源码，安装 RL 依赖（不动镜像自带的 torch / vLLM / triton），下载模型，生成数据 |
| `run_flowgrpo.sh` | 容器内 | 官方配方加 §4 的覆盖项；`LUMEN=1` 启用 §5 的补丁 |
| `lumen_rl_patch.py` | Ray worker 内 | 只在 actor 进程里安装 Lumen 补丁，并打印生效计数 |
| `compare.py` | 任意位置 | 验证 reward、训练 reward、分阶段耗时和吞吐，与官方参考对比 |
