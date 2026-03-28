# GRPO Training Analysis — NousResearch/Llama-2-70b-hf

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | NousResearch/Llama-2-70b-hf (70B parameters) |
| Parallelism | FSDP1 across 8 AMD MI GPUs |
| Training Steps | 30 |
| Micro Batch Size | 1 |
| Gradient Accumulation | 1 |
| Num Generations per Prompt | 8 |
| Max Completion Length | 256 tokens |
| Max Prompt Length | 512 tokens |
| Learning Rate | 5e-6 (linear decay) |
| Beta (KL penalty) | 0.0 |
| Gradient Checkpointing | Enabled |
| Dataset | trl-lib/Capybara |
| Total Training Time | 2h 05m (~250s/step avg) |

### Reward Function

Conciseness-based reward that penalises both trivially short and excessively
long responses, targeting a sweet spot around 30-60 words:

```
r(completion) =
    0.1                               if word_count < 5
    0.3 + 0.7 * word_count / 60       if 5 <= word_count <= 60
    max(0, 1.0 - (word_count-60)/120)  if word_count > 60
```

This produces variance across completions for the same prompt, which is the
requirement for GRPO to compute non-zero advantages and generate a policy
gradient signal.

---

## Key Findings

### 1. Reward vs. Step — Policy Learning Confirmed

![Reward vs Step](grpo_curves.png)

| Metric | Steps 1-5 | Steps 6-15 | Steps 16-30 |
|---|---|---|---|
| Mean Reward | 0.19 | 0.33 | 0.41 |
| Reward Std | 0.26 | 0.25 | 0.29 |

The mean reward rises from **0.11** (step 1) to a plateau around **0.4-0.6**
(steps 20-30), with peaks of **0.64** at steps 10 and 29. This confirms the
70B model is optimising its generation strategy in response to the reward
signal. The high variance reflects per-prompt diversity in the Capybara
dataset — some prompts naturally elicit longer or shorter responses.

### 2. Response Length vs. Step — Behavioural Adaptation

The most striking curve. Mean completion length drops from **225 tokens**
(step 1) to a range of **100-165 tokens** (steps 20-30):

| Phase | Mean Length (tokens) | Clipped Ratio |
|---|---|---|
| Steps 1-5 | 235 | 0.75 |
| Steps 6-15 | 190 | 0.55 |
| Steps 16-30 | 147 | 0.38 |

The model learns that shorter, more focused responses earn higher reward.
The clipped ratio (fraction of completions hitting max_length) drops from
75% to 38%, confirming the model is terminating responses earlier rather
than generating until truncation.

### 3. Entropy vs. Step — KL Divergence Proxy

With `beta=0.0`, no explicit KL penalty is applied and no reference model is
maintained. Entropy serves as a proxy: lower entropy indicates the policy has
diverged further from the initial (higher-entropy) model.

| Phase | Mean Entropy |
|---|---|
| Steps 1-5 | 1.90 |
| Steps 6-15 | 1.83 |
| Steps 16-30 | 2.03 |

Entropy dips during steps 10-18 (the most active learning phase, where
gradient norms also spike), then partially recovers as the learning rate
decays and the policy stabilises. The recovery in entropy at later steps
suggests the model explores again once gradients become smaller — a healthy
sign that it hasn't mode-collapsed.

### 4. Loss vs. Step — Policy Gradient Dynamics

Loss oscillates around zero with occasional large negative excursions:

| Step | Loss | Grad Norm | Event |
|---|---|---|---|
| 15 | -0.56 | 11.7 | Model discovers short-response strategy (48 tokens mean, reward 0.53) |
| 18 | -1.60 | 38.3 | Aggressive update — completions drop to 17 tokens mean |
| 20 | 0.02 | 9.0 | Recovery after over-correction |
| 25 | -0.51 | 4.3 | Refined short-response strategy (150 tokens, reward 0.61) |
| 30 | -0.36 | 18.0 | Final step — still actively optimising |

Negative loss in GRPO means the policy is increasing probability of
high-reward completions relative to low-reward ones within each group.
The gradient norm spikes at steps 15 and 18 represent the model
"discovering" that shorter completions yield higher reward — a phase
transition in the learned policy.

### 5. Win Rate (over baseline)

Win rate is the fraction of recent steps (rolling window of 5) where the
mean reward exceeds the initial baseline (mean reward of steps 1-3 =
0.165).  The first 3 steps are the warmup period (win rate = 0).

| Phase | Win Rate |
|---|---|
| Steps 1-3 | 0.0 (warmup) |
| Steps 4-7 | 0.5 → 1.0 (policy learning) |
| Steps 8-14 | 0.6-1.0 (stabilising) |
| Steps 15-30 | 0.6-1.0 (converged) |

The win rate **rises** from 0.0 to 1.0 and remains near 1.0 for the
second half of training.  This confirms the model reliably outperforms its
initial policy.  Brief dips to 0.6 (steps 11-15) correspond to steps
where the model over-shortened responses below the reward sweet spot.

### 6. Training Stability

The previous 10-step run with a constant reward function
(`reward_fn = 1.0`) failed to produce any training signal:

| Metric | Constant Reward (10 steps) | Conciseness Reward (30 steps) |
|---|---|---|
| Loss | 0.0 (all steps) | 0.27 avg, -1.60 to 0.48 range |
| Grad Norm | 0.0 (all steps) | 5.2 avg, 2.1 to 38.3 range |
| Mean Reward | 1.0 (all steps) | 0.11 to 0.64 range |
| Response Length | 213 avg (no change) | 225 → 99 (clear trend) |

The NCCL ALLGATHER timeout from the earlier run (which crashed at step 2
of the first attempt) was resolved by reducing memory pressure:
`GRAD_ACCUM=1`, `MAX_COMPLETION_LENGTH=256`, and
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.

---

## Interpretation

The 30-step GRPO training of LLaMA-2-70B on 8 AMD MI GPUs demonstrates:

1. **Functional RL loop.** The GRPO algorithm successfully computes
   advantages from reward variance across grouped completions and backpropagates
   policy gradients through the 70B model under FSDP1.

2. **Behavioural change.** The model measurably shifts its generation
   strategy from long, verbose completions (~225 tokens) toward concise
   responses (~150 tokens), matching the reward function's incentive
   structure.

3. **Stable convergence.** After an active exploration phase (steps 10-20)
   with large gradient updates, the reward stabilises around 0.4-0.6 and
   response length plateaus around 100-165 tokens. Entropy recovers from
   its minimum, indicating the policy avoids mode collapse.

4. **Infrastructure readiness.** The Lumen + TRL + FSDP1 stack can train
   a 70B model across 8 GPUs at ~250s/step with no NCCL failures over 30
   steps (2+ hours continuous), given appropriate memory configuration.

---

## Files

| File | Description |
|---|---|
| `grpo_eval_log.jsonl` | Per-step metrics in JSONL format (30 entries) |
| `grpo_curves.png` | 6-panel regression plot (reward, entropy, length, win rate, loss, grad norm) |
| `ANALYSIS.md` | This document |
