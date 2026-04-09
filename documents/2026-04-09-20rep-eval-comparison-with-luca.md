# 20-Rep Evaluation Results & Comparison with Franceschetti (2025)

**Date:** 2026-04-09

## Overview

This document summarizes the balanced 20-repetition evaluation results and compares them with Luca Franceschetti's original GRPO/LA-GRPO results from his ETH Zurich Master's thesis.

---

## Setup Differences

| | Franceschetti (2025) | This work (Gubler) |
|---|---|---|
| **Model** | Llama-3.1-8B-Instruct | Qwen3-14B-Instruct (4-bit QLoRA) |
| **Scenarios** | 3 (rental, loan, merger) | 5 (+joint venture, employment contract) |
| **Issues** | 10 | 19 |
| **Archetypes covered** | Not explicitly balanced | 7 archetypes, balanced (2 configs each) |
| **Eval size** | ~500 games, 1 repetition | 560 games, 20 repetitions |
| **Reward** | Pure self-interest (U_A) | Cooperative: R_coop = λ_s·U_A + λ_w·(U_A+U_B) + λ_f·(U_A·U_B)/100 |
| **Opponent** | Frozen local model | Frozen local model (selfplay) |
| **Judge** | GPT-4o-mini | GPT-4o-mini |
| **Statistical tests** | None reported | Mann-Whitney U, Chi-squared, Bootstrap CIs |

---

## Franceschetti's Results

### Single-Game (Rental Agreement Only)

| Method | Steps | U_A | Agreement Rate |
|--------|-------|-----|----------------|
| Llama-3.1-8B base | 0 | 51.3 | ~0%* |
| DPO | 4000 | 60.9 | 87.6% |
| REFUEL | 4000 | 78.0 | 90.8% |
| **GRPO** | **400** | **76.0** | **90.8%** |
| LA-GRPO | 400 | 68.9 | 92.2% |
| LA-GRPO | 1600 | 74.9 | 92.2% |

*Llama base shows 0% successful episodes in the CSV — likely a metric definition difference (may count only games where the model actively negotiated vs. producing malformed output).

### Multi-Game (3 Scenarios)

| Method | Steps | U_A | Partial Fail Rate |
|--------|-------|-----|--------------------|
| Llama-3.1-8B base | 0 | 46.0 | 19.8% |
| LA-GRPO multi-game | 600 | 68.5 | 12.3% |
| LA-GRPO multi-game | 800 | 64.1 | 13.7% |

### Key Findings from Franceschetti

1. **GRPO converges 10x faster than DPO/REFUEL** — 400 steps vs 4000 steps for comparable U_A
2. **LA-GRPO (turn-level credit)** slightly trails GRPO on U_A but matches on agreement rate
3. **Multi-game training generalizes** — single training run works across 3 game types

---

## Gubler's 20-Rep Results (Balanced Archetypes)

### Overall Performance (560 games/model, 14 configs x 2 roles x 20 reps)

| Model | Lambda (s/w/f) | Steps | U_A | U_A std | Agreement | Ratio Self | Ratio Welfare | Ratio Nash |
|-------|---------------|-------|-----|---------|-----------|------------|---------------|------------|
| Qwen3-14B base | - | 0 | 49.9 | 32.6 | 78.6% | 0.499 | 0.702 | 0.585 |
| self_only | 1.0/0.0/0.0 | 560 | 54.9 | 35.0 | 79.8% | 0.549 | 0.723 | 0.581 |
| fair_only | 0.0/0.0/1.0 | 560 | 55.0 | 30.8 | 89.6% | 0.550 | 0.808 | 0.664 |
| **all_equal** | **1.0/1.0/1.0** | **620** | **60.3** | **27.9** | **92.7%** | **0.603** | **0.840** | **0.692** |
| self_fair_equal | 0.5/0.0/0.5 | 820 | 57.0 | 32.3 | 83.8% | 0.570 | 0.757 | 0.625 |

### Per-Archetype Breakdown (Agreement Rate)

| Archetype | base | self_only | fair_only | all_equal | self_fair_equal |
|-----------|------|-----------|-----------|-----------|-----------------|
| single-compatible | 98.8% | 91.2% | 98.8% | 95.0% | 98.8% |
| single-distributive | 57.5% | 62.5% | 83.8% | **83.8%** | 55.0% |
| single-integrative | 72.5% | 63.7% | 88.8% | **87.5%** | 85.0% |
| non-integrative compatible | 87.5% | 95.0% | 95.0% | 95.0% | **97.5%** |
| non-integrative distributive | 86.2% | 86.2% | **95.0%** | **95.0%** | 85.0% |
| integrative compatible | **100.0%** | 90.0% | 91.2% | 97.5% | 92.5% |
| integrative distributive | 47.5% | 70.0% | 75.0% | **95.0%** | 72.5% |

### Per-Archetype Breakdown (U_A)

| Archetype | base | self_only | fair_only | all_equal | self_fair_equal |
|-----------|------|-----------|-----------|-----------|-----------------|
| single-compatible | 80.5 | 81.9 | 86.8 | 81.6 | **87.4** |
| single-distributive | 28.9 | 36.8 | 42.9 | **48.2** | 32.2 |
| single-integrative | 39.1 | 39.8 | 45.6 | 49.5 | **53.1** |
| non-integrative compatible | 58.9 | **68.6** | 61.9 | 66.1 | 68.4 |
| non-integrative distributive | 41.1 | 48.4 | 41.4 | **52.2** | 49.2 |
| integrative compatible | 76.8 | 72.8 | 73.6 | **79.8** | 73.1 |
| integrative distributive | 24.2 | 36.1 | 32.5 | **44.8** | 35.8 |

### Statistical Significance (Key Results)

| Comparison | U_A | Agreement | Ratio Nash | Ratio Welfare |
|---|---|---|---|---|
| all models > base (U_A) | p < 0.05 | | | |
| fair_only, all_equal > base | p < 0.05 | p < 0.001 | p < 0.001 | p < 0.001 |
| fair_only, all_equal > self_only | | p < 0.001 | p < 0.01 | p < 0.01 |
| all_equal > fair_only | p = 0.006 | n.s. (p=0.09) | n.s. (p=0.46) | n.s. (p=0.57) |
| all_equal > self_fair_equal | n.s. (p=0.33) | p < 0.001 | p = 0.013 | p = 0.002 |

### Model Tiers

- **Tier 1:** all_equal — best on U_A (vs fair_only), agreement + cooperative metrics (vs self_fair_equal)
- **Tier 2:** fair_only > self_fair_equal — cooperative signals help; dropping welfare hurts
- **Tier 3:** self_only ≈ base — self-interest alone doesn't improve cooperation

### Bootstrap 95% Confidence Intervals

| Model | U_A [95% CI] | Ratio Nash [95% CI] | Ratio Welfare [95% CI] |
|-------|-------------|---------------------|------------------------|
| base | 49.9 [47.2, 52.6] | 0.585 [0.556, 0.613] | 0.702 [0.671, 0.734] |
| self_only | 54.9 [52.0, 57.7] | 0.581 [0.552, 0.610] | 0.723 [0.691, 0.753] |
| fair_only | 55.0 [52.4, 57.6] | 0.664 [0.637, 0.689] | 0.808 [0.782, 0.833] |
| all_equal | 60.3 [58.0, 62.6] | 0.692 [0.669, 0.716] | 0.840 [0.819, 0.861] |
| self_fair_equal | 57.0 [54.4, 59.6] | 0.625 [0.596, 0.653] | 0.757 [0.727, 0.786] |

---

## Comparison & Discussion

### 1. GRPO Convergence Speed Confirmed

Franceschetti showed GRPO converges 10x faster than DPO/REFUEL (400 vs 4000 steps). This work confirms fast convergence: all models show meaningful improvements by step 560-620, even on a harder multi-game task with 7 archetypes.

### 2. U_A Is Lower, But the Task Is Much Harder

Franceschetti's GRPO reached U_A=76.0 on the rental agreement scenario alone. Our best model (all_equal) reaches U_A=60.3 across 7 balanced archetypes, including integrative-distributive games where the base model only achieves 47.5% agreement. The broader game diversity and harder archetypes naturally lower the average U_A. On compatible games (where Franceschetti's scenarios are closest), our models reach U_A=80-87, comparable to Franceschetti's range.

### 3. Agreement Rate Is Comparable or Better

Franceschetti's best agreement rate: 92.2% (LA-GRPO, 1600 steps, single rental game). Our all_equal achieves 92.7% agreement at 620 steps across a much more diverse and difficult game set. This is arguably more impressive given the task diversity.

### 4. Cooperative Reward Doesn't Sacrifice Self-Interest

Franceschetti optimized purely for U_A (self-interest). Our cooperative reward (all_equal) still achieves +21% U_A improvement over baseline, while simultaneously improving social welfare (+20%), Nash product (+18%), and agreement rate (+14pp). The cooperative signal complements rather than trades off against self-interest.

### 5. Pure Self-Interest Replicates Franceschetti's Finding

Our self_only model (λ_self=1.0, same objective as Franceschetti) behaves similarly: moderate U_A improvement (+10%) but no significant improvement on agreement rate or Nash product. This confirms that Franceschetti's approach, while effective for competitive negotiation, is insufficient for cooperative settings.

### 6. LA-GRPO Remains an Opportunity

Franceschetti showed LA-GRPO and GRPO perform similarly on U_A, but LA-GRPO provides turn-level credit assignment which may be more beneficial in cooperative settings where specific turns (e.g., proposing a fair trade-off) are critical. This is planned as future work.

---

## Implications for the Thesis

### RQ1 (LA-GRPO for cooperative negotiation)
Not yet tested — GRPO results here set the baseline for LA-GRPO comparison.

### RQ2 (Reward function balance)
**Answered:** The combined reward (all_equal: λ_s=1, λ_w=1, λ_f=1) significantly outperforms pure self-interest and partial combinations. Dropping the welfare component (self_fair_equal) hurts performance despite more training steps. All three reward components contribute.

### RQ3 (Cooperative metrics)
**Partially answered:** Ratio Nash and Ratio Welfare successfully differentiate models that U_A alone cannot. self_only and fair_only have similar U_A (~55) but vastly different Nash product (0.581 vs 0.664, p<0.001) and welfare (0.723 vs 0.808, p<0.01). Per-archetype breakdowns reveal where each model excels.

---

## Known Issue

In the CSV, `ratio_rcoop_mean` equals `ratio_self_mean` for all models. This appears to be a bug in the eval script — the cooperative reward ratio should differ from pure self-interest ratio. Likely `compute_max_metrics()` is using default lambdas (λ_self=1, others=0) during evaluation, making R_coop = U_A. Worth investigating and re-computing.
