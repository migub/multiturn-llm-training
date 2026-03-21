# Temperature Change: 0.6 → 1.0 — Mode Collapse Fix

## Problem: Mode Collapse

At step 360 of the self-only run (`eunj3abz`), all 8 GRPO generations for the same prompt produce nearly identical dialogues:

```
Sample 1: "Counteroffer of 3 days of remote work and a $3500 training budget."  → reward=80
Sample 2: "Counteroffer of 3 days of remote work and a $3500 training budget."  → reward=0
Sample 3: "Counteroffer of 3 days of remote work and a $3500 training budget."  → reward=0
Sample 4: "Counteroffer of 3 days of remote work and a $4000 training budget."  → reward=80
Sample 5: "Counteroffer of 3 days of remote work and a $4000 training budget."  → reward=60
Sample 6: "Counteroffer of 3 days of remote work and a $4000 training budget."  → reward=0
Sample 7: "Counteroffer of 4 days of remote work and a $4500 training budget."  → reward=0
Sample 8: "Counteroffer of 3 days of remote work and a $3500 training budget."  → reward=0
```

The agent has converged to a single strategy: always counter with 3 days remote + $3500-4000 training budget. The 8 samples are nearly identical.

## Why This Is Bad for GRPO

GRPO works by generating G completions for the same prompt, comparing their rewards, and reinforcing the best ones relative to the group mean. This requires **diversity** — if all 8 samples are the same, there's nothing to compare, and the advantage signal is noise.

With identical samples:
- Reward variance within the group is near zero
- Advantages ≈ 0 for all samples
- No meaningful gradient signal → training stalls

## Root Cause: Low Temperature

Qwen3-14B-Instruct has a default `temperature=0.6` in its `generation_config`. This was not explicitly set in our training script, so the model's default was used.

At temperature=0.6, the probability distribution over next tokens is quite peaked — the model strongly favors its top choice. After some training reinforces a particular strategy, temperature=0.6 isn't enough randomness to explore alternatives.

Additionally, the **opponent** uses the same low temperature, so it always responds the same way too — further reducing dialogue diversity.

## Fix: Temperature = 1.0

Changed the default temperature to 1.0 via a new `--temperature` CLI argument. This:
- Flattens the token probability distribution → more diverse completions
- Ensures the 8 GRPO samples actually differ → meaningful advantage signal
- Allows the agent to explore different negotiation strategies

Temperature 1.0 is the standard for GRPO/PPO training. Lower temperatures are for inference (when you want consistent, high-quality output), not for training (when you need exploration).

## Code Change

Added `--temperature` flag to `grpo_single_gpu.py` (default 1.0), passed to `GRPOConfig(temperature=args.temperature)`. This flows through to both the agent's and opponent's generation configs.

## Usage

```bash
# Default (1.0, recommended for training)
python multiturn_llm_training/grpo/grpo_single_gpu.py --use-wandb ...

# Explicit
python multiturn_llm_training/grpo/grpo_single_gpu.py --temperature 1.0 --use-wandb ...
```
