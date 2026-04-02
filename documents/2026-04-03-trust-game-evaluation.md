# Trust Game Evaluation

**Date:** 2026-04-03  
**Purpose:** Out-of-domain evaluation to test whether cooperative negotiation training generalizes to a different game-theoretic setting.

---

## What is the Trust Game?

A two-player sequential game that cleanly separates selfish from cooperative behavior:

- **Investor** starts with 10 points, sends 0-10 to the Trustee
- Sent amount is **tripled** (x3) before the Trustee receives it
- **Trustee** decides how much of the tripled amount to return (0 to 3x sent)

**Payoffs:**
```
Investor payoff = (10 - sent) + returned
Trustee payoff  = (3 x sent) - returned
```

**Key reference points:**

| Outcome | Send | Return | Investor | Trustee | Social Welfare | Nash Product |
|---------|------|--------|----------|---------|----------------|--------------|
| Nash equilibrium (selfish) | 0 | 0 | 10 | 0 | 10 | 0 |
| Pareto optimum (cooperative) | 10 | 15 | 15 | 15 | 30 | 225 |

A purely self-interested Trustee always keeps everything (return 0), so a rational Investor sends nothing. Cooperative behavior requires the Trustee to reciprocate, which in turn requires the Investor to trust.

---

## Why This Evaluation?

The models are trained on multi-issue negotiation (rental agreements, joint ventures, etc.). The Trust Game tests whether the learned cooperative behavior **transfers** to a structurally different game:

- Different format: 2-turn sequential decisions instead of multi-round negotiation
- Different actions: numerical amounts instead of negotiated agreement text
- Different structure: one player has all the power (Trustee can exploit Investor)

**Expected results:**
- **Self-only models** (lambda_fair=0) should converge toward Nash equilibrium — low send, low return
- **Cooperative models** (lambda_fair>0) should show higher trust and reciprocation — approaching Pareto

This speaks to RQ4 (robustness/generalization) and provides evidence for whether cooperative reward shaping produces genuinely cooperative agents vs agents that only learned negotiation-specific strategies.

---

## Implementation

**Script:** `evaluations/trust_game_eval.py`

Single standalone file (~280 lines). No new dependencies beyond what the project already uses (torch, transformers, peft, bitsandbytes).

### How It Works

1. **Load model** — Base model (Qwen3-14B-Instruct) in 4-bit quantization + LoRA adapter from checkpoint
2. **Play N games per role** — Trained model plays as Investor (N games) and Trustee (N games)
3. **Opponent** — Frozen base model (LoRA disabled), same pattern as training
4. **Extract actions** — Regex-based extraction of send/return amounts (no GPT-4o-mini needed)
5. **Compute metrics** — Payoffs, social welfare, nash product, return ratio
6. **Report** — Console output + optional JSON file with full results

### Dialogue Flow

```
Turn 1: Investor generates response  (e.g., "I send 8 points")
Turn 2: Trustee receives Investor's message, generates response  (e.g., "I return 12 points")
```

The system prompts instruct models to include "I send X points" / "I return X points" to make regex extraction reliable. Qwen3 `<think>` tags are stripped automatically.

### Adapter Switching

Same pattern as training (`multiturn_grpo_trainer.py`):
- `model.enable_adapter_layers()` — trained model (LoRA active)
- `model.disable_adapter_layers()` — frozen base model (opponent)

When the trained model plays Investor, adapters are ON for the Investor turn and OFF for the Trustee turn (and vice versa).

---

## Usage

```bash
# Base model baseline (no LoRA)
python evaluations/trust_game_eval.py --checkpoint none --num-games 50

# Evaluate a trained checkpoint
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo_balanced_0.5_0.5/checkpoint-100 \
    --num-games 50

# Save results to JSON for later analysis
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo_balanced_0.5_0.5/checkpoint-100 \
    --num-games 50 \
    --output-json evaluations/trust_results/balanced_0.5_0.5.json
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to LoRA checkpoint dir, or `none` for base model |
| `--model-name` | `OpenPipe/Qwen3-14B-Instruct` | Base model name |
| `--num-games` | 50 | Number of games per role |
| `--temperature` | 0.6 | Sampling temperature |
| `--output-json` | None | Path to save JSON results |
| `--seed` | 42 | Random seed |
| `--verbose` | (flag) | Print full dialogues and payoffs for each game |
| `--no-quantized` | (flag) | Disable 4-bit quantization |

### Hardware Requirements

- **GPU required** (BitsAndBytes 4-bit needs CUDA)
- **~9-10 GB VRAM** for Qwen3-14B in 4-bit
- **~5-10 minutes** for 50 games per role on RTX 5090

---

## Metrics

Per role (trained-as-investor, trained-as-trustee):

| Metric | Description | Selfish | Cooperative |
|--------|-------------|---------|-------------|
| `avg_send` | Points sent by Investor (0-10) | 0 | 10 |
| `avg_return` | Points returned by Trustee | 0 | 15 |
| `avg_return_ratio` | return / (3 x sent) | 0% | 50% |
| `avg_investor_payoff` | Investor's final points | 10 | 15 |
| `avg_trustee_payoff` | Trustee's final points | 0 | 15 |
| `avg_social_welfare` | Sum of both payoffs | 10 | 30 |
| `avg_nash_product` | Product of both payoffs | 0 | 225 |
| `parse_failure_rate` | % of games where extraction failed | — | — |
| `std_*` | Standard deviations for all numeric metrics | — | — |

---

## Output Format

### Console (default)

Per-game one-liner during play, then summary:

```
Playing 50 games as INVESTOR...
  Game   1/50: send=8, return=12  [OK] (2.3s)
  Game   2/50: send=10, return=15  [OK] (1.9s)
  ...

============================================================
TRUST GAME EVALUATION RESULTS
============================================================
  Checkpoint:    output/grpo_balanced_0.5_0.5/checkpoint-100
  Games/role:    50
  Temperature:   0.6
------------------------------------------------------------
  Reference — Nash equilibrium:  send=0,  return=0   -> (10, 0)  SW=10  NP=0
  Reference — Pareto optimum:    send=10, return=15  -> (15, 15) SW=30  NP=225
------------------------------------------------------------

  [Trained as INVESTOR]  (valid: 48/50)
    Avg send:            7.42 / 10    (std: 2.31)
    Avg return:          11.30          (std: 4.12)
    Return ratio:        50.8%
    ...

  [Trained as TRUSTEE]  (valid: 47/50)
    ...
```

### Console (with `--verbose`)

Shows full dialogues and payoffs for each game:

```
Playing 50 games as INVESTOR...
  Game   1/50: send=8, return=12  [OK] (2.3s)
    Investor: I'll send 8 points to show good faith. You'll receive 24...
    Trustee:  Thank you for the trust. I return 12 points to you...
    Payoffs:  Investor=14, Trustee=12
```

### JSON (with `--output-json`)

```json
{
  "args": { "checkpoint": "...", "num_games": 50, ... },
  "metrics": {
    "investor": { "avg_send": 7.42, "avg_return": 11.3, ... },
    "trustee": { ... }
  },
  "games": {
    "investor": [
      {
        "investor_text": "I'll send 8 points to show trust...",
        "trustee_text": "Thank you. I return 12 points...",
        "send": 8, "return_amt": 12, "parse_ok": true
      },
      ...
    ],
    "trustee": [ ... ]
  }
}
```

The JSON includes raw dialogues for every game, useful for qualitative analysis or cherry-picking examples for the thesis.

---

## Suggested Evaluation Plan

Run for each trained checkpoint + base model baseline:

```bash
# Base model
python evaluations/trust_game_eval.py --checkpoint none \
    --output-json evaluations/trust_results/base_model.json

# Self-only (lambda_self=1.0, lambda_fair=0.0)
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo_self_only/checkpoint-100 \
    --output-json evaluations/trust_results/self_only.json

# Fair-only (lambda_self=0.0, lambda_fair=1.0)
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo_fair_only/checkpoint-100 \
    --output-json evaluations/trust_results/fair_only.json

# Balanced (lambda_self=0.5, lambda_fair=0.5)
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo_balanced/checkpoint-100 \
    --output-json evaluations/trust_results/balanced_0.5_0.5.json
```

Compare social welfare and nash product across configs to measure cooperation transfer.
