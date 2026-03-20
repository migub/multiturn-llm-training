# Code Review & Fixes — 2026-03-20

Session reviewing `grpo_single_gpu.py`, `env.py`, `evaluator.py`, and wandb run `vmizj47a`.

---

## Bugs Found & Fixed

### 1. `lookup_payoff` substring matching bug (CRITICAL)
**File:** `evaluator/evaluator.py`
**Problem:** The partial match logic `label_clean in value_clean` caused `"0%"` to match `"70%"` (since `"0%"` is a substring of `"70%"`). This returned payoff 0 for any percentage-based agreement. All cooperative scenarios with `%` labels (JV data-sharing, revenue-split, etc.) were broken — valid agreements got reward 0.
**Fix:** Changed to a two-pass approach: exact match first, then partial match picking the **longest matching label** instead of the first match.
**Impact:** All previous runs with percentage-based labels had corrupted rewards.

### 2. `loss_type` defaulting to DAPO instead of GRPO
**File:** `multiturn_llm_training/grpo/grpo_single_gpu.py`
**Problem:** `loss_type` was not explicitly set in `GRPOConfig`. Recent TRL versions changed the default from `"grpo"` to `"dapo"` (Decoupled Alignment Policy Optimization). DAPO removes the KL penalty and uses asymmetric clipping. This explains why `clip_ratio` was always 0 in training runs.
**Fix:** Added `loss_type="grpo"` to training args.
**Impact:** All previous runs used DAPO loss instead of GRPO.

### 3. `room_price.yaml` payoff bug
**File:** `envs/negotiation/configs/issues/room_price.yaml`
**Problem:** Side B (buyer) payoffs were `[0,1,2,...,9]` (ascending) — same direction as Side A. This made the buyer get more points for paying more money, which is wrong for a distributive issue. Also had a typo: `"$000"` instead of `"$800"`.
**Fix:** Changed Side B payoffs to `[9,8,7,...,0]` (descending) and fixed labels to ascending `["$100"..."$1000"]` for both sides.

### 4. Wandb metrics only logged last batch (not averaged)
**File:** `envs/negotiation/env.py`
**Problem:** `wandb.log(metrics, commit=False)` was called every step, but only the last call before the trainer's `commit=True` (every `logging_steps`) survived. Metrics from intermediate steps were overwritten and lost.
**Fix:** Added an accumulator that collects metrics across steps and logs the average every `logging_steps` calls, then resets.
**Impact:** Metrics are now averaged over `logging_steps * num_generations` samples instead of just `num_generations`.

### 5. Failed evaluations mixed into ratio averages
**File:** `envs/negotiation/env.py`
**Problem:** When evaluation fails (no agreement), all metrics are set to 0.0 and included in the average. This conflates agreement rate with negotiation quality — `ratio_self_mean=0.4` could mean either 40% payoff on all games or 80% payoff on 50% of games.
**Fix:** Added `negotiation/agreed/*` metrics that only average over successful negotiations. Both the original (all-inclusive) and agreed-only metrics are now logged.

### 6. `gen-ra-duration` + `gen-ra-duration-distributive` combo allowed
**File:** `envs/negotiation/env.py`
**Problem:** These two issues represent the same negotiation topic (lease duration) but with different payoff structures. They could be combined in a 2-issue game, which doesn't make sense.
**Fix:** Added `excluded_combos` filter to skip this combination when generating game configs.

---

## Configuration Changes

### `logging_steps` default: 20 -> 5
**File:** `multiturn_llm_training/grpo/grpo_single_gpu.py`
More frequent logging for better training visibility.

### `save_steps` now defaults to `eval_steps`
**File:** `multiturn_llm_training/grpo/grpo_single_gpu.py`
Checkpoints are saved at the same steps as evaluations, so each eval has a corresponding checkpoint. Default `save_steps` changed from 200 to `None` (auto-set to `eval_steps`).

### Fixed eval dataset
**File:** `envs/negotiation/env.py`
Added `create_eval_dataset()` method with a curated, deterministic set of 10 samples per game type:
- **multi-game:** rental (single), loan (2-issue), merger (2-issue), JV compat+dist, EC integrative+compat
- **cooperative-only:** JV single dist, JV compat+dist, JV compat+integrative, EC single dist, EC integrative+compat
- **out-of-domain:** rio copa — single compat, single integrative, integrative+dist, compat+integrative, integrative+integrative

Previous eval used random permutation + random weights, making results non-comparable across runs.

### Added `cooperative-only` game type
**File:** `envs/negotiation/env.py`
New game type for ablation studies (RQ2). Uses only joint-venture and employment-contract scenarios. Accessible via `--game-type cooperative-only`.

---

## Observations from Run `vmizj47a`

### Agent reveals payoff information
The agent (and frozen local opponent) leak internal payoff values in dialogue, e.g. "R&D budget of $8M (payoff 40)". The system prompt says not to do this, but the model ignores it. This gives an unfair signal that won't exist against real opponents.

### `ratio_welfare` always 1.0 for pure distributive games
In distributive games, `U_A + U_B = constant` regardless of outcome. So `ratio_welfare` is always 1.0 — it only provides signal for integrative/compatible games.

### U_A consistently lower than U_B
Across the run, the trained agent (U_A) gets worse deals than the opponent (U_B). Expected since the opponent plays its pretrained policy without learning pressure.

### Gap in training steps 58-73
15 steps missing from logs. Likely caused by the long eval runtime (~42 min for 10 samples) or a crash/recovery.

### `max_completion_length=200` may be too tight
Average completion is 100-140 tokens. With 5 rounds of dialogue, 200 tokens total can truncate longer negotiations before agreement is reached.

### Partial agreements get nonzero reward (not fixed yet)
The game rules say partial agreements should give payoff 0, but `get_payoffs()` sums whatever issues are agreed. In 2-issue games where only 1 issue is resolved, the agent still gets a positive reward. This is a potential fix for a future session.
