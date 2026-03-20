# vLLM Colocate Mode Implementation — 2026-03-20

## What was added

Support for `--use-vllm` flag in `grpo_single_gpu.py` that enables vLLM colocate mode for faster agent generation. The existing non-vLLM training path is completely unchanged.

## Files changed

### New: `multiturn_llm_training/grpo/multiturn_rollout.py`

A `rollout_func` for TRL's `GRPOTrainer` that handles multi-turn dialogue generation with vLLM.

- `create_multiturn_rollout()` — factory function that returns the rollout function with configured parameters (max_rounds, max_tokens_per_turn, opponent_model)
- Agent turns use `vllm_llm.chat()` — all 8 dialogues batched in a single call per round
- Opponent turns use `model.generate()` with LoRA disabled (same logic as `MultiTurnGRPOTrainer`)
- `_tokenize_and_mask()` — tokenizes the dialogue and builds `env_mask` (1=agent, 0=opponent), same logic as `_tokenize_conversation()` in the existing trainer
- `_local_generate()` — single-turn generation for opponent, same as `_gen_response()` in existing trainer
- `_openai_response()` — OpenAI API opponent, same as existing
- Returns `conversations` as an extra field so the reward function receives structured dialogue data

### Modified: `multiturn_llm_training/grpo/grpo_single_gpu.py`

- Added `--use-vllm` flag (default: False)
- Added `--vllm-gpu-memory` flag (default: 0.3)
- When `--use-vllm` is set:
  - Uses base `GRPOTrainer` (not `MultiTurnGRPOTrainer`) with `rollout_func`
  - Configures vLLM colocate mode (`vllm_mode="colocate"`, `vllm_enable_sleep_mode=False`)
  - Builds a `prompt_2` mapping from the dataset so the rollout function can look up the opponent's system prompt
- When `--use-vllm` is NOT set: identical to before (uses `MultiTurnGRPOTrainer`)

### Modified: `envs/negotiation/env.py`

- Reward function now accepts `conversations` kwarg (from rollout's extra fields)
- When `conversations` is provided, uses it instead of `completions`
- When not provided (non-vLLM path), behavior is unchanged

### NOT modified: `multiturn_llm_training/grpo/multiturn_grpo_trainer.py`

Completely untouched. Remains the fallback for non-vLLM runs.

## How it works

### Without vLLM (existing path)

```
grpo_single_gpu.py
  → MultiTurnGRPOTrainer (custom subclass)
    → _generate_and_score_completions()
      → _play_negotiation() → model.generate() for each turn (sequential)
      → _tokenize_conversation() → assistant_mask
      → compute logprobs, rewards, advantages, loss (custom overrides)
```

### With vLLM (new path)

```
grpo_single_gpu.py
  → GRPOTrainer (base class) + rollout_func
    → rollout_func()
      → vllm_llm.chat() for agent turns (batched)
      → model.generate() with LoRA off for opponent turns (sequential)
      → _tokenize_and_mask() → env_mask
      → returns prompt_ids, completion_ids, env_mask, conversations
    → GRPOTrainer handles logprobs, rewards, advantages, loss (built-in)
```

## Usage

```bash
# With vLLM
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-vllm \
    --vllm-gpu-memory 0.3 \
    --use-wandb \
    --game-type multi-game \
    --model-name OpenPipe/Qwen3-14B-Instruct \
    --run-name grpo_vllm_test

# Without vLLM (unchanged)
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type multi-game \
    --model-name OpenPipe/Qwen3-14B-Instruct \
    --run-name grpo_no_vllm
```

## Requirements

- `pip install vllm` on the training server
- Sleep mode is disabled (`vllm_enable_sleep_mode=False`) so both vLLM and the training model stay in VRAM — needed for local opponent with LoRA disable/enable
- A100 80GB recommended. RTX 5090 32GB may be tight with a 14B model.

## Not yet tested

This implementation has not been tested end-to-end. Recommended first step: run with `--test --use-vllm` to verify the integration works before a full training run.
