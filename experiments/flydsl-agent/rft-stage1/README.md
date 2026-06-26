# RFT Stage A — Diversity-Preserving Rejection Fine-Tuning

基于 SFT v5e 模型，大规模生成候选 kernel → FlyDSL-Gym 沙箱验证 → 保留所有编译通过实现 → 短 SFT 训练。

## 结果

| 指标 | SFT v5e (baseline) | **RFT v1** | Target |
|------|-------------------|-----------|--------|
| Overall API Score | 74.1% | **74.6%** | >60% ✅ |
| Sandbox Compile | 21.9% (42/192) | **30.7% (59/192)** | >10% ✅ |
| L4 (Advanced) | 49% | **54%** | >50% ✅ |
| Operator Coverage | 11/12 | **12/12** | ✅ |

HuggingFace: [Zhangdanyang/Qwen2.5-Coder-RFT-v1](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v1)

## 训练流程

### 前置条件

- SFT v5e 已完成，merged model 在 `/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5e`
- FlyDSL-Gym 沙箱镜像 `flydsl-gym:latest` 已构建
- RL spec 在 `/home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl`

### Step 1: 候选生成 (~17h)

用 v5e 模型对 gfx950 spec 生成候选 kernel 代码。

```bash
docker run --rm --init \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    -v /path/to/experiments:/workspace/experiments \
    -v /path/to/sft-model:/model:ro \
    -v /path/to/dataset:/data:ro \
    -v /path/to/rft-results:/results \
    lumen/flydsl-cpt:latest \
    python3 /workspace/experiments/rft-stage1/generate_candidates.py \
        --model /model \
        --specs /data/data/rl/train-00000-of-00001.jsonl \
        --output /results/candidates_rft_full.jsonl \
        --n-candidates 16 --max-specs 100 --hardware gfx950 --device cuda:0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-specs` | 100 | 从 213 个 gfx950 spec 中按算子均匀采样 |
| `--n-candidates` | 16 | 每 spec 生成 16 个候选 |
| `--hardware` | gfx950 | 只用 gfx950 spec |
| temperature | 0.8 | 鼓励多样性 |
| top_p | 0.95 | |
| 3 种 prompt 风格 | precise / natural / optimization | 轮换使用 |

实际采样 84 个 spec → 1,344 个候选。

### Step 2: 沙箱验证 (~20min)

对全部候选做静态检查 + Docker 沙箱编译验证。

```bash
python3 verify_candidates.py \
    --input /path/to/candidates_rft_full.jsonl \
    --output /path/to/verified_rft_full.jsonl \
    --metadata /path/to/verify_stats_rft_full.json \
    --use-sandbox
```

三级过滤：
1. **语法检查** — `compile(code, 'exec')` + FlyDSL pattern + 非平凡 (>15 行)
2. **沙箱编译** — Docker 容器中真实 FlyDSL 编译
3. **多样性** — edit distance ratio < 0.9 去重

结果：1,344 → 922 静态通过 → **326 沙箱通过** (24.3%)

### Step 3: 构建 RFT 数据集

将沙箱验证通过的候选转为 ChatML 格式，与 v5e SFT 数据合并。

```bash
python3 build_rft_dataset.py \
    --verified /path/to/verified_rft_full.jsonl \
    --sft-data /path/to/sft/train-00000-of-00001.jsonl \
    --output /path/to/rft_train.jsonl \
    --rft-repeat 2
```

| 数据 | 数量 |
|------|------|
| SFT v5e 原始数据 | 3,889 |
| RFT 验证通过 ×2 | 652 |
| **合并总数** | **4,541** |
| RFT 占比 | 14.4% |

关键设计：**Diversity-preserving** — 保留所有通过编译的实现，不取 top-K。
同一 spec 可能有多种不同实现（不同 tiling / pipeline / import 模式），全部保留。

### Step 4: RFT 训练 (~1.2h)

在 v5e merged model 上做 1 epoch 短训练。

```bash
# 用环境变量覆盖超参，复用 SFT trainer
source config_rft.sh
# 训练脚本与 SFT 相同 (train_sft.py)，只是 base model 和数据不同
```

| 参数 | 值 | 说明 |
|------|-----|------|
| Base model | Qwen2.5-Coder-SFT-v5e | merged model，非 base |
| LR | 5e-6 | 比 SFT (1e-5) 低，保护已有能力 |
| Epochs | 1 | 短训练，避免过拟合到 RFT 数据 |
| MAX_STEPS | 568 | 4541 / GBS=8 × 1 epoch |
| LoRA | r=64, alpha=128 | 与 SFT 相同 |
| Dropout | 0.05 | 比 SFT (0.1) 低 |
| val_loss | 0.970 → 0.967 | 轻微改善 |

### Step 5: 导出 + 评估

```bash
# 导出 HF
python3 export_hf.py \
    --base-model /path/to/Qwen2.5-Coder-SFT-v5e \
    --dcp-path /path/to/rft-results/final/final \
    --output /path/to/Qwen2.5-Coder-RFT-v1 \
    --lora-rank 64 --lora-alpha 128

# Benchmark (25 题 API Score)
python3 eval_sft.py \
    --base-model /path/to/base \
    --sft-model /path/to/Qwen2.5-Coder-RFT-v1 \
    --output benchmark_rft_v1.json

# 沙箱评估 (20 spec × 16 候选)
python3 generate_candidates.py --model /path/to/RFT-v1 --max-specs 20 ...
python3 verify_candidates.py --use-sandbox ...
```

## 全自动 Pipeline

一键执行全部步骤：

```bash
bash /home/danyzhan/rft-results/run_rft_pipeline.sh
# 日志: /home/danyzhan/rft-results/rft_pipeline.log
```

总耗时约 20h（生成 17h + 验证 0.5h + 训练 1.2h + 评估 3h）。

## 文件清单

```
rft-stage1/
├── README.md                  # 本文件
├── REPORT.md                  # 实验报告（详细结果分析）
├── generate_candidates.py     # 候选生成器
├── verify_candidates.py       # 静态 + 沙箱验证
├── build_rft_dataset.py       # verified → ChatML + 合并
├── config_rft.sh              # 训练超参
├── run_rft.sh                 # (旧) 单步脚本
└── results/
    ├── benchmark_rft_v1.json          # 25 题 benchmark
    ├── verify_stats_rft_full.json     # 1344 候选沙箱统计
    ├── verify_stats_rft_eval.json     # RFT 模型 192 候选评估
    ├── candidates_v5e_gfx950.jsonl    # v5e 评估候选
    └── rft_pipeline.log               # 完整 pipeline 日志
```

## 关键数据路径

| 数据 | 路径 |
|------|------|
| 大规模候选 (84 specs) | `/home/danyzhan/rft-results/candidates_rft_full.jsonl` |
| 沙箱通过候选 | `/home/danyzhan/rft-results/verified_rft_full.jsonl` |
| RFT 训练数据 | `/home/danyzhan/rft-results/rft_train.jsonl` |
| RFT 模型 (merged) | `/home/danyzhan/rft-results/Qwen2.5-Coder-RFT-v1` |
| DCP checkpoint | `/home/danyzhan/rft-results/final/` |
