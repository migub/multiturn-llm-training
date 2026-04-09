# Negotiation Evaluation Results (Balanced Archetypes, 20 Repetitions)

**Date:** 2026-04-09
**Eval script:** `evaluations/run_negotiation_eval.py`
**Setup:** 560 games per model (14 configs x 2 roles x 20 repetitions), selfplay with frozen base opponent, GPT-4o-mini judge, temperature=1.0, seed=42, max 5 rounds per negotiation.

## Eval Dataset

14 configurations covering all 7 archetypes (2 configs each), played from both starting roles = 28 unique games per repetition.

| # | Archetype | Scenario | Issues | Weights (A/B) |
|---|-----------|----------|--------|---------------|
| 1 | single-distributive | Rental | rent | [1]/[1] |
| 2 | single-distributive | EC | salary | [1]/[1] |
| 3 | single-compatible | JV | R&D budget | [1]/[1] |
| 4 | single-compatible | EC | training budget | [1]/[1] |
| 5 | single-integrative | EC | remote work | [1]/[1] |
| 6 | single-integrative | JV | decision authority | [1]/[1] |
| 7 | non-integrative distributive | EC | salary + equity | [50,50]/[50,50] |
| 8 | non-integrative distributive | Loan | rate + duration | [50,50]/[50,50] |
| 9 | non-integrative compatible | Loan | amount + rate | [50,50]/[50,50] |
| 10 | non-integrative compatible | Merger | benefits + ownership | [50,50]/[50,50] |
| 11 | integrative distributive | JV | revenue split + decision auth | [70,30]/[30,70] |
| 12 | integrative distributive | EC | salary + remote work | [70,30]/[30,70] |
| 13 | integrative compatible | EC | remote work + training budget | [70,30]/[30,70] |
| 14 | integrative compatible | JV | R&D budget + data sharing | [70,30]/[30,70] |

## Results

| Model | Lambda (s/w/f) | Steps | Agreement | U_A mean | U_A std | Ratio Self | Ratio Welfare | Ratio Nash |
|-------|---------------|-------|-----------|----------|---------|------------|---------------|------------|
| base_model | - | 0 | 78.6% | 49.9 | 32.6 | 0.499 | 0.702 | 0.585 |
| self_only | 1.0 / 0.0 / 0.0 | 560 | 79.8% | 54.9 | 35.0 | 0.549 | 0.723 | 0.581 |
| fair_only | 0.0 / 0.0 / 1.0 | 560 | 89.6% | 55.0 | 30.8 | 0.550 | 0.808 | 0.664 |
| all_equal | 1.0 / 1.0 / 1.0 | 620 | **92.7%** | **60.3** | 27.9 | **0.603** | **0.840** | **0.692** |
| self_fair_equal | 0.5 / 0.0 / 0.5 | 820 | 83.8% | 57.0 | 32.3 | 0.570 | 0.757 | 0.625 |

## Per-Archetype Breakdown

### Agreement Rate by Archetype

| Archetype | base | self_only | fair_only | all_equal | self_fair_equal |
|-----------|------|-----------|-----------|-----------|-----------------|
| single-compatible | 98.8% | 91.2% | 98.8% | 95.0% | 98.8% |
| single-distributive | 57.5% | 62.5% | 83.8% | **83.8%** | 55.0% |
| single-integrative | 72.5% | 63.7% | 88.8% | **87.5%** | 85.0% |
| non-integrative compatible | 87.5% | 95.0% | 95.0% | 95.0% | **97.5%** |
| non-integrative distributive | 86.2% | 86.2% | **95.0%** | **95.0%** | 85.0% |
| integrative compatible | **100.0%** | 90.0% | 91.2% | 97.5% | 92.5% |
| integrative distributive | 47.5% | 70.0% | 75.0% | **95.0%** | 72.5% |

### U_A by Archetype

| Archetype | base | self_only | fair_only | all_equal | self_fair_equal |
|-----------|------|-----------|-----------|-----------|-----------------|
| single-compatible | 80.5 | 81.9 | 86.8 | 81.6 | **87.4** |
| single-distributive | 28.9 | 36.8 | 42.9 | **48.2** | 32.2 |
| single-integrative | 39.1 | 39.8 | 45.6 | 49.5 | **53.1** |
| non-integrative compatible | 58.9 | **68.6** | 61.9 | 66.1 | 68.4 |
| non-integrative distributive | 41.1 | 48.4 | 41.4 | **52.2** | 49.2 |
| integrative compatible | 76.8 | 72.8 | 73.6 | **79.8** | 73.1 |
| integrative distributive | 24.2 | 36.1 | 32.5 | **44.8** | 35.8 |

### Ratio Nash by Archetype

| Archetype | base | self_only | fair_only | all_equal | self_fair_equal |
|-----------|------|-----------|-----------|-----------|-----------------|
| single-compatible | 0.661 | 0.741 | 0.771 | 0.705 | **0.776** |
| single-distributive | 0.415 | 0.451 | 0.602 | 0.564 | 0.347 |
| single-integrative | 0.626 | 0.444 | 0.753 | **0.769** | 0.709 |
| non-integrative compatible | 0.710 | **0.787** | 0.715 | 0.758 | 0.778 |
| non-integrative distributive | 0.759 | 0.690 | **0.802** | **0.858** | 0.755 |
| integrative compatible | 0.706 | 0.648 | 0.697 | **0.764** | 0.685 |
| integrative distributive | 0.215 | 0.309 | 0.307 | **0.428** | 0.324 |

## Key Findings

### 1. All trained models improve over baseline
Every trained model significantly outperforms the base model on U_A (p < 0.05). The improvement ranges from +10% (self_only) to +21% (all_equal).

### 2. all_equal is the best overall model
The model trained with lambda_self=1.0, lambda_welfare=1.0, lambda_fair=1.0 achieves the highest scores:
- **U_A = 60.3** (vs 49.9 baseline, +21%)
- **Agreement rate = 92.7%** (vs 78.6% baseline)
- **Ratio Nash = 0.692** (vs 0.585 baseline)
- **Ratio Welfare = 0.840** (vs 0.702 baseline)

### 3. Self-interest alone is insufficient
self_only (lambda_self=1.0 only) improves U_A marginally but does NOT improve agreement rate, ratio_nash, or ratio_welfare significantly over baseline. The cooperative reward signals (welfare, fairness) are what drive meaningful improvement.

### 4. Fairness is a strong individual signal
fair_only significantly outperforms self_only on agreement rate (p < 0.001), ratio_nash (p < 0.001), and ratio_welfare (p < 0.01). This suggests fairness-based rewards are more effective than pure self-interest for learning cooperative negotiation.

### 5. all_equal dominates on the hardest archetype
The integrative-distributive archetype (where trade-offs exist but no free wins) is the most challenging. all_equal achieves 95% agreement vs 47.5% baseline — nearly doubling it. This is where the combined reward signal shows its greatest advantage.

### 6. self_fair_equal underperforms all_equal
Despite additional training steps (820 vs 620), self_fair_equal (lambda_self=0.5, lambda_welfare=0.0, lambda_fair=0.5) is significantly worse than all_equal on agreement rate (p < 0.001) and ratio_welfare (p < 0.01). Dropping the welfare component hurts performance.
