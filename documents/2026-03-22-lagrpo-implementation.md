# LA-GRPO Implementation (Turn-Level Credit Assignment)

**Date:** 2026-03-22
**Context:** Implementing Luca's LA-GRPO algorithm in Michael's single-GPU trainer

---

## What is LA-GRPO?

Standard GRPO generates G independent dialogues per prompt, computes a reward for each, and uses group-relative advantages to train. The problem: **every agent token in the dialogue gets the same advantage signal**. If a 5-round negotiation fails because of a bad move in round 4, round 1's tokens are equally "blamed."

LA-GRPO (Local Advantage GRPO) fixes this with **turn-level sampling**:

1. Sample a random turn `h` from a geometric distribution
2. Play rounds 0..h-1 **once** (shared prefix — identical across all G generations)
3. From round h onward, generate **G independent continuations**
4. Compute rewards and advantages as usual
5. Only apply loss to agent tokens from turn h onward

Since all variation in reward comes from decisions at turn h+, the advantage signal provides **local credit assignment** — the model learns which turn-level decisions matter.

---

## How the Geometric Distribution Works

```
h ~ Geometric(p=0.3), bounded by max_rounds - 1
```

With 5 rounds (h in {0, 1, 2, 3, 4}):

| h | Approx. Probability | Meaning |
|---|---------------------|---------|
| 0 | ~30% | No prefix, fully independent (= standard GRPO) |
| 1 | ~21% | Share round 0, branch from round 1 |
| 2 | ~15% | Share rounds 0-1, branch from round 2 |
| 3 | ~10% | Share rounds 0-2, branch from round 3 |
| 4 | ~24% (remaining mass) | Share rounds 0-3, only last round varies |

Early turns are sampled more frequently (more variation, more learning signal). The bounded geometric ensures all turns get coverage.

**Mean h ≈ 2.3** — on average, about half the dialogue is shared prefix and half is independently generated.

---

## Implementation Details

### Files Changed

**`multiturn_grpo_trainer.py`:**
- Added `sample_geometric_bounded()` utility (from Luca's code)
- Added `turn_level_sampling` and `turn_sampling_p` parameters to `__init__`
- Added `_play_prefix()` method — plays h rounds and returns state for continuation
- Modified `_play_negotiation()` — accepts `resume_state` to continue from shared prefix
- Modified `_tokenize_conversation()` — accepts `mask_from_agent_turn` to only mark agent tokens from turn h onward in `assistant_mask`
- Modified `_generate_and_score_completions()` — samples h, generates shared prefix once, then G independent continuations

**`grpo_single_gpu.py`:**
- Added `--turn-level-sampling` flag
- Added `--turn-sampling-p` flag (default 0.3)

### Generation Flow

```
Standard GRPO (turn_level_sampling=False):
  For each of G generations:
    Play full 5-round dialogue independently
    Mask all agent tokens for loss

LA-GRPO (turn_level_sampling=True, e.g. sampled_h=2):
  1. Play rounds 0-1 once (shared prefix)
  2. For each of G generations:
       Resume from prefix state
       Play rounds 2-4 independently
       Mask only agent tokens from round 2+ for loss
```

### What Happens at h=0?

When h=0 (sampled ~30% of the time), there is no shared prefix. All G generations are fully independent — identical to standard GRPO. The mask covers all agent tokens. This means LA-GRPO naturally includes standard GRPO as a special case.

### Performance Benefit

With h > 0, the shared prefix saves generation time:
- h=2 with G=8: instead of 8 × 10 turns = 80 forward passes, it's 1 × 4 + 8 × 6 = 52 forward passes (~35% reduction)
- Average savings with Geometric(0.3): ~20-25% fewer forward passes per step

### Eval Mode

During evaluation, LA-GRPO is disabled — always uses standard GRPO with full independent dialogues. This ensures eval metrics are comparable across runs.

---

## Usage

```bash
# Standard GRPO (default, no change)
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type multi-game \
    --run-name grpo_standard

# LA-GRPO
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type multi-game \
    --turn-level-sampling \
    --run-name lagrpo_multigame

# LA-GRPO with custom p (higher p = more early-turn sampling)
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --use-wandb \
    --game-type multi-game \
    --turn-level-sampling \
    --turn-sampling-p 0.5 \
    --run-name lagrpo_p05
```

---

## Comparison: Standard GRPO vs LA-GRPO

| Aspect | Standard GRPO | LA-GRPO |
|--------|--------------|---------|
| Generations per prompt | G independent dialogues | Shared prefix + G continuations |
| Advantage granularity | Same advantage for all agent tokens | Advantage only for tokens after turn h |
| Credit assignment | Whole-dialogue | Turn-level (local) |
| Forward passes per step | G × 2 × max_rounds | prefix + G × 2 × remaining_rounds |
| Eval behavior | Standard | Standard (LA-GRPO disabled) |
| When h=0 | N/A | Identical to standard GRPO |

---

## Relationship to RQ1

**RQ1: Does LA-GRPO improve learning in cooperative negotiation?**

The hypothesis: cooperative negotiation requires nuanced multi-turn strategies (e.g., start competitive to anchor, then make concessions to build trust). Standard GRPO can't distinguish which turns drove the outcome. LA-GRPO can.

**Ablation for RQ1:**

| Run | Method | Compare |
|-----|--------|---------|
| A | Standard GRPO (self-only) | Baseline |
| B | LA-GRPO (self-only) | Does turn-level credit help even without cooperative reward? |
| C | Standard GRPO (cooperative lambdas) | Does cooperative reward help without credit assignment? |
| D | LA-GRPO (cooperative lambdas) | Full system — does combining both give the best result? |

If D > C and B > A, LA-GRPO provides value. If D > B and C > A, cooperative reward provides value. If D >> max(B, C), they're complementary.
