# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Master Thesis:** "From Competition to Cooperation – Teaching LLMs Pareto-Efficient Negotiation with LA-GRPO"  
**Author:** Michael Gubler, Lucerne University of Applied Sciences (MScIDS)  
**Supervisor:** Oliver Staubli (HSLU), Co-Supervisor: Peter Niederberger (Tincan AG)  
**Deadline:** 29 May 2026  
**Language:** Michael speaks German and English. Code is in English, discussions can be in German.

### Goal

Train LLMs to negotiate **cooperatively** (not just competitively) using reinforcement learning. The agent should balance self-interest with collective welfare, achieving Pareto-efficient outcomes where both parties benefit. The key innovation is a cooperative reward function: `R_coop = λ_self × U_A + λ_welfare × (U_A + U_B) + λ_fair × (U_A × U_B) / 100`.

### Based On

This work extends [Franceschetti (2025)](https://github.com/lfranceschetti/multiturn-llm-training) – "Using the Advantage: Teaching LLMs to Negotiate in Multi-Turn Dialogues with Local Advantage GRPO" (ETH Zurich Master's Thesis). Luca's work focused on zero-sum bargaining; Michael's extends it to cooperative/mixed-motive negotiation.

### Research Questions

- **RQ1:** Does LA-GRPO improve learning in cooperative negotiation? (Turn-level credit assignment)
- **RQ2:** How should reward functions balance self-interest vs. collective welfare? (λ parameter ablation)
- **RQ3:** Which metrics capture cooperative success? (Social welfare, Nash product, Pareto efficiency)
- **RQ4:** Can cooperative agents remain robust against adversarial opponents?
- **RQ5:** Does cooperation training affect general LLM capabilities? (MMLU-Pro, GLUE, IFEval)

---

## Repository Structure

### Michael's Negotiation Repo (fork of Luca's)

```
multiturn-llm-training/
├── envs/negotiation/
│   ├── env.py                    ← MODIFIED: Cooperative reward + archetype tracking + wandb metrics + ratio logging
│   ├── games.py                  ← Luca's original (Game, Issue classes)
│   └── configs/
│       ├── games/
│       │   ├── generic-rental-agreement.yaml   (Luca)
│       │   ├── generic-loan-agreement.yaml     (Luca)
│       │   ├── generic-merger.yaml             (Luca)
│       │   ├── joint-venture.yaml              ← NEW (Michael)
│       │   └── employment-contract.yaml        ← NEW (Michael)
│       ├── issues/
│       │   ├── gen-ra-*.yaml, gen-la-*.yaml, gen-m-*.yaml  (Luca's 10 issues)
│       │   ├── jv-rd-budget.yaml               ← NEW compatible
│       │   ├── jv-revenue-split.yaml           ← NEW distributive
│       │   ├── jv-data-sharing.yaml            ← NEW compatible
│       │   ├── jv-decision-authority.yaml      ← NEW integrative
│       │   ├── ec-salary.yaml                  ← NEW distributive
│       │   ├── ec-remote-work.yaml             ← NEW integrative
│       │   ├── ec-training-budget.yaml         ← NEW compatible
│       │   ├── ec-equity.yaml                  ← NEW distributive
│       │   └── ec-project-scope.yaml           ← NEW compatible
│       └── general_game_rules.yaml
├── evaluator/
│   ├── evaluator.py              ← MODIFIED: Added lookup_payoff() for text-based labels
│   ├── openai_model.py           (Luca)
│   ├── evaluation_outcome.txt    (Luca)
│   ├── evaluation_outcome_single.txt  (Luca)
│   └── evaluation_outcome_multi.txt   (Luca)
├── multiturn_llm_training/
│   ├── train.py                  ← Luca's DPO/REFUEL entry point
│   └── grpo/
│       ├── grpo_single_gpu.py    ← Michael's training script (entry point)
│       ├── grpo.py               (Luca's original, multi-GPU + vLLM)
│       └── lagrpo_trainer.py     (Luca's LA-GRPO trainer, multi-GPU + vLLM)
├── secrets.json                  ← OpenAI API key (not in git)
└── CLAUDE.md                     ← This file
```

### Michael's TRL Fork (separate repo)

```
trl/ (forked from huggingface/trl, branch: master_thesis)
└── trainer/
    ├── grpo_trainer.py               ← Original TRL (unchanged)
    └── grpo_trainer_multiturn.py     ← MICHAEL'S FILE: GRPOTrainer with multi-turn support
```

Install with: `pip install git+https://github.com/migub/trl.git@master_thesis`

---

## Common Commands

### Single-GPU Training (Michael's setup)

```bash
# Cooperative multi-game training
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type multi-game \
    --lambda-self 1.0 \
    --lambda-welfare 0.5 \
    --lambda-fair 0.3 \
    --num-generations 8 \
    --train-size 200 \
    --opponent-model gpt-4o-mini \
    --run-name grpo_cooperative_v1

# Baseline (self-interest only, frozen local opponent)
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type generic-rental-agreement \
    --num-generations 8 \
    --run-name grpo_baseline_rental

# Quick test
python multiturn_llm_training/grpo/grpo_single_gpu.py --test

# Resume from checkpoint
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --resume-from-checkpoint output/grpo_multiturn/checkpoint-50 \
    [other args...]
```

### Luca's Original Commands (multi-GPU, not used by Michael)

```bash
# Start vLLM servers (requires 2+ GPUs)
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <model_name>
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model <model_name>

# GRPO/LA-GRPO
CUDA_VISIBLE_DEVICES=2,3 accelerate launch multiturn_llm_training/grpo/grpo.py

# DPO/REFUEL (Hydra config, DeepSpeed)
accelerate launch multiturn_llm_training/train.py --config-path configs_training/experiment --config-name <config>
```

### Environment Setup (new server)

```bash
git clone https://github.com/migub/multiturn-llm-training.git
cd multiturn-llm-training
pip install --upgrade torch transformers accelerate peft bitsandbytes datasets
pip install unsloth hydra-core omegaconf pyyaml openai retry attrs wandb numpy pandas
pip install git+https://github.com/migub/trl.git@master_thesis --force-reinstall --no-deps
export OPENAI_API_KEY="your-key"
echo '{"openai": {"api_key": "your-key"}}' > secrets.json
wandb login
apt update && apt install tmux -y
```

### Tests

```bash
python tests/games.py
python tests/grpo.py
python tests/masking.py
```

---

## Architecture

### Two Training Pipelines

**Michael's single-GPU pipeline (active):**

- Model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit quantized, ~5GB VRAM)
- LoRA: r=8, alpha=16, dropout=0.1, target_modules=[q,k,v,o,gate,up,down]\_proj (0.26% trainable)
- No vLLM: direct `model.generate()` for dialogue generation
- Opponent: frozen local model (LoRA disabled) OR OpenAI API (gpt-4o-mini)
- Reference model: LoRA disabled = base model (no separate model needed, no extra VRAM)
- GPU: A100 80GB or RTX 5090 32GB

**Luca's multi-GPU pipeline (original, not actively used):**

- 4 GPUs: 2× vLLM servers (policy + opponent) + 2× training
- Custom TRL fork with vLLM multi-server communication
- Uses accelerate + DeepSpeed

### Training Method Differences

| Method  | Loss                               | Sampling                            | Pipeline                 |
| ------- | ---------------------------------- | ----------------------------------- | ------------------------ |
| DPO     | Sigmoid preference loss            | Full trajectory                     | Offline                  |
| REFUEL  | Quadratic advantage loss           | Turn-level                          | Offline                  |
| GRPO    | Group relative policy optimization | G trajectories per prompt           | Online (Michael's focus) |
| LA-GRPO | GRPO + turn-level sampling         | G continuations from sampled turn h | Online (future work)     |

---

## Key Modified Files — Detailed

### `envs/negotiation/env.py` — Cooperative Reward

**New parameters:** `lambda_self`, `lambda_welfare`, `lambda_fair` control the cooperative reward balance.

**`compute_max_metrics()`:** Brute-forces all possible outcomes (max 121 combinations) to find maximum achievable U_A, social_welfare, nash_product, and R_coop. Used for normalized ratio logging so training curves are comparable across game types.

**Wandb logging:** Logs raw metrics (U_A, U_B, social_welfare, nash_product, agreement_rate) and four normalized ratios (ratio_self, ratio_welfare, ratio_nash, ratio_rcoop) directly from the reward function.

**Archetype tracking:** Each sample includes `archetype` field from `get_archetype_from_game()`.

### `evaluator/evaluator.py` — Text Label Support

**`lookup_payoff()`:** Before numeric extraction, tries exact/partial text matching against payoff labels. Enables labels like "full scope" alongside "$1200". Falls back to numeric interpolation if no match.

### `trl/trainer/grpo_trainer_multiturn.py` — Multi-Turn GRPO

Modified copy of TRL's `grpo_trainer.py`. Changes marked with `# MULTITURN:` comments.

**New parameters:** `multiturn=True`, `max_negotiation_rounds`, `max_tokens_per_turn`, `opponent_model`

**New methods:**

- `_multiturn_generate_single_response()` → one message via model.generate()
- `_play_negotiation()` → full dialogue, LoRA on/off for agent/opponent, or OpenAI API
- `_tokenize_multiturn_conversation()` → creates assistant_mask (1=agent, 0=opponent), with fallback
- `_generate_multiturn()` → returns results as tool_mask for seamless integration with `_compute_loss()`
- `_openai_opponent_response()` → OpenAI API for opponent

**How it integrates:** `_generate()` checks `self.multiturn` flag → calls `_generate_multiturn()` → returns assistant_mask as `tool_mask_list` → TRL's existing `mask = completion_mask * tool_mask` handles the rest.

### `grpo_single_gpu.py` — Training Script

CLI entry point. Key args: `--game-type`, `--lambda-self/welfare/fair`, `--num-generations`, `--opponent-model`, `--max-rounds`, `--max-tokens-per-turn`, `--use-wandb`, `--test`, `--resume-from-checkpoint`.

---

## Game Archetypes

| Archetype                | Description                                      | Two-Issue Count |
| ------------------------ | ------------------------------------------------ | --------------- |
| Compatible               | Both prefer same outcome                         | 2 (7%)          |
| Mixture                  | Mix of distributive + compatible                 | 20 (69%)        |
| Integrative-Distributive | All contested but different weights → trade-offs | 7 (24%)         |

5 scenarios, 19 issues, 48 game combinations (Luca: 3 scenarios, 10 issues, 23 games).

---

## Known Issues

1. **agent_token_ratio=0 → NaN crash:** `_tokenize_multiturn_conversation()` sometimes fails. Fixed with fallback marking all tokens as agent.
2. **LLaMA 8B weak instruction following:** Invents issues, reveals payoffs. Improves with training.
3. **OpenAI opponent too strong:** Untrained agent gets stuck. Train with frozen local first.
4. **Slow sequential generation:** 80 forward passes per step (~2-4 min). Could batch 8 dialogues.
5. **Text-based labels:** Fixed with `lookup_payoff()` but partial matches can still fail.
6. **Reward not comparable across games:** Fixed with normalized ratios (0-1).

---

## Wandb Metrics

### TRL defaults

- `train/reward`, `train/reward_std` — raw R_coop (noisy across game types)
- `train/kl` — KL from reference (<5 is ok)
- `train/clip_ratio/*` — PPO clipping (<0.3 is ok)
- `train/step_time` — seconds per step

### Custom metrics (env.py → wandb.log)

- `negotiation/U_A_mean`, `negotiation/U_B_mean` — raw payoffs
- `negotiation/social_welfare_mean` — U_A + U_B
- `negotiation/nash_product_mean` — U_A × U_B
- `negotiation/agreement_rate` — fraction reaching agreement
- **`negotiation/ratio_self_mean`** — U_A / max(U_A), 0-1, game-normalized
- **`negotiation/ratio_welfare_mean`** — welfare / max(welfare), 0-1
- **`negotiation/ratio_nash_mean`** — nash / max(nash), 0-1
- **`negotiation/ratio_rcoop_mean`** — R_coop / max(R_coop), 0-1 ← **best metric for training progress**

### Trainer metrics

- `multiturn/agent_token_ratio` — fraction agent tokens (~0.3-0.5)
- `multiturn/archetype/*` — game archetype counts per step

---

## Next Steps

1. **Early stopping** — stop dialogue when agreement reached (saves tokens/time)
2. **Batch generation** — all 8 agent turns simultaneously (~8x faster generation)
3. **Ablation studies** — vary λ_self, λ_welfare, λ_fair for RQ2
4. **LA-GRPO** — turn-level credit assignment (Luca's key contribution)
5. **Evaluation pipeline** — fixed test set, Pareto frontiers per archetype
6. **Robustness** — cooperative agent vs adversarial opponents (RQ4)
7. **Benchmarks** — MMLU-Pro, GLUE, IFEval for capability preservation (RQ5)

---

## Michael's Preferences

- Limited formal math background → use analogies and simplified explanations
- Learns by running code and observing results
- Prefers interactive JSX tutorials with dark theme, step-by-step code walkthroughs
- German for explanations, English for code
- Completed Deep RL course: REINFORCE → Actor-Critic → PPO → RLHF → LoRA → DPO → GRPO → LA-GRPO
