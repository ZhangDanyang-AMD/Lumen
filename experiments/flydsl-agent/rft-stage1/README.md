# RFT Stage A — Diversity-Preserving Rejection Fine-Tuning

基于 SFT v5f 模型，大规模生成候选 kernel → FlyDSL-Gym 沙箱验证 (编译+运行+正确性) → 保留所有编译通过实现 → 1 epoch 训练。

## 结果

| 指标 | v5e | v5f SFT | **RFT v5f** | Target |
|------|-----|---------|-------------|--------|
| Overall API Score | 74% | 74% | **75%** | >60% |
| L5 (Expert) | 50% | 50% | **57%** | >20% |
| Format Compliance | — | 96% | **96%** | >90% |
| Sandbox Compile (122 specs) | 22% | 38% | **53%** | — |

**三层验证** (1952 candidates from RFT model):

| Level | Passed | Rate |
|-------|--------|------|
| Compilation | 944 | 48% |
| Runtime | 211 | 11% |
| Correctness | 0 | 0% |

Correctness 0% 说明 kernel 内部计算逻辑是错的 — 这是 RL Stage B/C 的目标。

HuggingFace: [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f)

详细分析见 [REPORT.md](REPORT.md)。

## 运行

### 前置条件

- SFT v5f model 在 `/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f`
- FlyDSL-Gym 沙箱镜像 `flydsl-gym:latest`
- Docker image `lumen/flydsl-cpt:latest`
- RL specs 在 `/home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl`
- 8x AMD MI350X GPUs

### 全自动 Pipeline

一键执行全部 7 步 (生成→验证→构建数据→训练→导出→评估)：

```bash
cd /home/danyzhan/Lumen/experiments/flydsl-agent/rft-stage1
bash run_rft.sh
```

总耗时约 26h (生成 23h + 验证 0.5h + 训练 2h + 导出+评估 1h)。

可通过环境变量自定义：

```bash
SFT_MODEL=/path/to/model \
MAX_SPECS=50 \
N_CANDIDATES=8 \
HARDWARE=gfx942 \
bash run_rft.sh
```

### 分步执行

#### Step 1: 候选生成 (~23h)

```bash
docker run --rm --init \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    -v experiments:/workspace/experiments \
    -v /path/to/sft-model:/model:ro \
    -v /path/to/dataset:/data:ro \
    -v /path/to/rft-results:/results \
    lumen/flydsl-cpt:latest \
    python3 /workspace/experiments/flydsl-agent/rft-stage1/generate_candidates.py \
        --model /model \
        --specs /data/data/rl/train-00000-of-00001.jsonl \
        --output /results/candidates_v5f_gfx950.jsonl \
        --n-candidates 16 --max-specs 213 --hardware gfx950 --device cuda:0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-specs` | 213 | 全部 gfx950 spec (实际采样 122 个) |
| `--n-candidates` | 16 | 每 spec 生成 16 个候选 |
| `--hardware` | gfx950 | 目标硬件 |
| temperature | 0.8 | 鼓励多样性 |
| 3 种 prompt 风格 | precise / natural / optimization | 轮换使用 |

#### Step 2-3: 沙箱验证 (~30min)

三层验证：静态检查 → Docker 沙箱编译 → 运行时执行 + 正确性检查

```bash
python3 verify_candidates.py \
    --input /path/to/candidates_v5f_gfx950.jsonl \
    --output /path/to/verified_v5f_gfx950.jsonl \
    --metadata /path/to/verify_stats_v5f_gfx950.json \
    --use-sandbox
```

验证统计输出 (`verify_stats_*.json`) 包含：
- `passed_static` — 语法 + FlyDSL pattern + ≥15 行
- `passed_sandbox` — FlyDSL JIT 编译通过
- `passed_runtime` — entry point 调用成功
- `passed_correctness` — 输出与 PyTorch reference 匹配
- `by_operator` — 每个算子的分项统计

#### Step 4: 构建 RFT 数据集

```bash
python3 build_rft_dataset.py \
    --verified /path/to/verified_v5f_gfx950.jsonl \
    --sft-data /path/to/format_aligned/train.jsonl \
    --output /path/to/rft_v5f_train.jsonl \
    --rft-repeat 2
```

关键设计：**Diversity-preserving** — 保留所有通过编译的实现，不取 top-K。

#### Step 5: RFT 训练 (~2h)

在 v5f merged model 上 1 epoch 训练。使用 `config_rft.sh` 超参。

| 参数 | 值 |
|------|-----|
| Base model | Qwen2.5-Coder-SFT-v5f (merged) |
| LR | 5e-6 |
| Epochs | 1 |
| LoRA | r=64, alpha=128, dropout=0.05 |

#### Step 6-7: 导出 + 评估

自动导出 HF 模型并运行 benchmark (API Score + format compliance + sandbox compilation)。

## 文件清单

```
rft-stage1/
├── README.md                  # 本文件
├── REPORT.md                  # 详细实验报告
├── generate_candidates.py     # 候选生成器 (N=16, 3 种 prompt 风格)
├── verify_candidates.py       # 静态 + 沙箱编译 + 运行时 + 正确性验证
├── build_rft_dataset.py       # verified → ChatML + 合并
├── config_rft.sh              # 训练超参 (lr=5e-6, 1 epoch)
└── run_rft.sh                 # 全自动 7 步 pipeline
```

## 关键数据路径

| 数据 | 路径 |
|------|------|
| v5f 候选 (122 specs) | `rft-results/candidates_v5f_gfx950.jsonl` |
| RFT 候选 (122 specs, RFT model) | `rft-results/candidates_rft_v5f_gfx950.jsonl` |
| 沙箱验证统计 (编译级) | `rft-results/verify_stats_rft_v5f_gfx950.json` |
| 三层验证统计 (含运行+正确性) | `rft-results/verify_stats_rft_v5f_runtime.json` |
| RFT 训练数据 | `rft-results/rft_v5f_train.jsonl` (8,275 条) |
| RFT 模型 (merged) | `rft-results/Qwen2.5-Coder-RFT-v5f` |
| Benchmark | `rft-results/benchmark_rft_v5f.json` |

## 沙箱验证器 (`sandbox/verify.py`)

三层验证：

1. **编译** — `importlib.util.spec_from_file_location` + `exec_module` 触发 FlyDSL JIT
2. **运行** — 检测 `@flyc.jit` / `launch_*` / `forward` 等 entry point，构造 operator-specific 输入，调用 kernel
3. **正确性** — 输出与 PyTorch reference 比较 (`torch.allclose`)

支持 12 个算子类型的输入构造和 reference 计算：gemm, softmax, rmsnorm, layernorm, rope, topk, quant, flash_attn, moe, mla, paged_attn, custom。
