# Implementation Plan: vLLM Colocate Mode for Multi-Turn Negotiation

## Current Bottleneck

The generation phase dominates training time (~238s per step in run `vmizj47a`). Currently, 8 dialogues × ~10 turns = ~80 sequential `model.generate()` calls per step. vLLM colocate mode can batch these calls, significantly reducing generation time.

## Key Design Decision: Use TRL's `rollout_func`

TRL's `GRPOTrainer` supports a `rollout_func` parameter — a custom function that handles generation and returns structured data. It supports an `env_mask` field that maps directly to our assistant mask concept. This means:

- **No need for `MultiTurnGRPOTrainer`** when using vLLM — the base `GRPOTrainer` handles logprobs, rewards, advantages, and loss natively
- Weight sync, sleep mode, and importance sampling are handled automatically
- The existing `MultiTurnGRPOTrainer` stays as a fallback for the non-vLLM path

## Step-by-Step Implementation

### Step 1: Create `multiturn_llm_training/grpo/multiturn_rollout.py` (new file)

Create a factory function that returns a rollout function with the correct signature:

```python
def create_multiturn_rollout(max_negotiation_rounds, max_tokens_per_turn, opponent_model):
    def multiturn_negotiation_rollout(prompts: list[str], trainer: GRPOTrainer) -> dict:
        # Access vLLM engine
        llm = trainer.vllm_generation.llm
        tokenizer = trainer.processing_class

        # Play dialogues turn-by-turn with BATCHED generation
        # Instead of 80 sequential model.generate() calls:
        # → ~10 batched llm.chat() calls (1 per round per speaker)

        for round in range(max_negotiation_rounds):
            # Batch ALL agent turns in a single llm.chat() call
            agent_prompts = [build_messages(dialogue_i) for i in range(batch_size)]
            agent_outputs = llm.chat(agent_prompts, sampling_params=SamplingParams(
                max_tokens=max_tokens_per_turn, temperature=1.0
            ))

            # Batch ALL opponent turns (OpenAI API)
            opponent_responses = [openai_call(msgs) for msgs in opponent_prompts]

        # Build return dict
        return {
            "prompt_ids": [...],       # tokenized system prompts
            "completion_ids": [...],   # tokenized full dialogues
            "logprobs": None,          # not needed initially
            "env_mask": [...],         # 1=agent token, 0=opponent token
        }

    return multiturn_negotiation_rollout
```

Key implementation details:
- Reuse tokenization/masking logic from `_tokenize_conversation()` in `multiturn_grpo_trainer.py`
- `env_mask` is automatically picked up by TRL as `tool_mask` in loss computation
- `vllm.SamplingParams` controls `max_tokens`, `temperature`, `top_p`, `top_k` per turn

### Step 2: Modify `grpo_single_gpu.py`

Add CLI flags and conditional trainer selection:

```python
parser.add_argument("--use-vllm", action="store_true", default=False)
parser.add_argument("--vllm-gpu-memory", type=float, default=0.3)
```

Conditional setup:

```python
if args.use_vllm:
    from multiturn_llm_training.grpo.multiturn_rollout import create_multiturn_rollout

    rollout_func = create_multiturn_rollout(
        max_negotiation_rounds=args.max_rounds,
        max_tokens_per_turn=args.max_tokens_per_turn,
        opponent_model=args.opponent_model,
    )

    training_args = GRPOConfig(
        ...,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        vllm_enable_sleep_mode=True,  # offload vLLM weights during training
    )

    trainer = GRPOTrainer(  # base class, NOT MultiTurnGRPOTrainer
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        rollout_func=rollout_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
else:
    # Existing path — no changes
    trainer = MultiTurnGRPOTrainer(...)
```

### Step 3: Adapt reward function interface (`env.py`)

The rollout function can return extra fields (e.g., `conversations`) that flow through to reward functions via `reward_kwargs`. Minor change to the reward function to accept structured conversation data from this path in addition to the current `completions` parameter.

### Step 4: No changes to `multiturn_grpo_trainer.py`

Kept as the fallback for `--use-vllm` not set. The existing non-vLLM training path remains fully functional.

## Files Summary

| File | Action |
|------|--------|
| `multiturn_llm_training/grpo/multiturn_rollout.py` | **New** — rollout function with batched vLLM generation |
| `multiturn_llm_training/grpo/grpo_single_gpu.py` | Add `--use-vllm`, `--vllm-gpu-memory` flags, conditional trainer |
| `envs/negotiation/env.py` | Minor: accept conversations from `reward_kwargs` |
| `multiturn_llm_training/grpo/multiturn_grpo_trainer.py` | **No changes** (fallback path) |

## Limitation: Local Opponent Not Supported with vLLM

In colocate mode, `sync_weights()` merges LoRA into the base weights before pushing to vLLM. There's no way to disable LoRA for opponent turns. Therefore:

- **vLLM mode requires `--opponent-model gpt-4o-mini`** (OpenAI API opponent)
- **Non-vLLM mode** still supports both local and OpenAI opponents

## Potential Challenges

1. **VRAM pressure:** vLLM colocate + training model + LoRA on single GPU. Sleep mode helps. A100 80GB should be fine; RTX 5090 32GB may be tight.

2. **BitsAndBytes compatibility:** Check that installed vLLM version supports 4-bit quantized models. TRL's `VLLMGeneration._init_vllm()` has detection logic for this.

3. **Chat template:** vLLM's `llm.chat()` applies the chat template internally. Ensure the tokenizer's template is compatible.

4. **Logprobs:** Initially return `logprobs=None` (disables importance sampling correction). Can add later by extracting per-token logprobs from vLLM output.

## Expected Speedup

The main gain is batching: instead of 80 sequential forward passes, we do ~10 batched `llm.chat()` calls (5 rounds × 2 speakers). vLLM's continuous batching and optimized kernels provide additional speedup on top of the batching itself.

---

## Addendum: Local Opponent IS Possible with vLLM Colocate

The limitation described above (requiring OpenAI API opponent) can be avoided. In the `rollout_func`, we have access to `trainer.model` — the full PeftModel with LoRA adapters. We can use a hybrid approach:

### Hybrid Generation Strategy

- **Agent turns:** use vLLM (`trainer.vllm_generation.llm.chat()`) — fast, batched, LoRA merged
- **Opponent turns:** use `trainer.model.generate()` with LoRA disabled — sequential, but the opponent only generates 1 response per dialogue (not 8), so speed is less critical

```python
# Agent turn — fast, batched via vLLM
agent_outputs = trainer.vllm_generation.llm.chat(agent_prompts, sampling_params)

# Opponent turn — regular model with LoRA off (base model = frozen opponent)
trainer.model.disable_adapter_layers()
opponent_outputs = trainer.model.generate(**opponent_inputs)
trainer.model.enable_adapter_layers()
```

### Why This Works

- The **agent's generation is the bottleneck** — GRPO samples 8 completions per prompt, each with ~5 turns = ~40 agent forward passes. vLLM batches these into ~5 calls.
- The **opponent generates only 1 response** per dialogue per round. Sequential `model.generate()` is fine here since we don't backprop through the opponent anyway.
- LoRA disable/enable is instant (no weight copying).

### Requirement: Disable Sleep Mode

This approach requires `vllm_enable_sleep_mode=False`, because sleep mode offloads the training model during generation — making `trainer.model.generate()` unavailable for opponent turns. Without sleep mode, both vLLM and the training model stay in VRAM simultaneously.

**VRAM impact:** Both models share the same base weights (vLLM has a copy with LoRA merged, training model has base + LoRA adapters separately). On A100 80GB this should be fine. On RTX 5090 32GB it may be tight with a 14B model.

### Updated Recommendation

- **A100 80GB:** Use hybrid approach (vLLM agent + local opponent, sleep mode off)
- **RTX 5090 32GB:** Either use sleep mode + OpenAI opponent, or test if memory fits without sleep mode
- **OpenAI opponent** remains the simplest option regardless of GPU
