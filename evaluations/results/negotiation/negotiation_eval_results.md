# Negotiation Evaluation Results (Multi-Game Selfplay)

**Date:** 2026-04-08
**Eval script:** `evaluations/run_negotiation_eval.py`
**Setup:** 10 games per model (5 eval configs x 2 starting roles), selfplay with frozen base opponent, GPT-4o-mini judge, temperature=1.0, seed=42, max 5 rounds per negotiation.

## Results

| Model | Lambda (s/w/f) | Steps | Agreement | U_A mean | U_A std | Social Welfare | Ratio Self | Ratio Welfare | Ratio Nash | Ratio R_coop |
|-------|---------------|-------|-----------|----------|---------|----------------|------------|---------------|------------|--------------|
| base_model | - | 0 | 70% | 44.1 | 33.0 | 88.2 | 0.441 | 0.591 | 0.499 | 0.441 |
| self_only | 1.0 / 0.0 / 0.0 | 560 | 70% | 50.1 | 36.2 | 95.4 | 0.501 | 0.609 | 0.523 | 0.501 |
| fair_only | 0.0 / 0.0 / 1.0 | 560 | 80% | 53.9 | 29.6 | 106.3 | 0.539 | 0.709 | 0.611 | 0.539 |
| all_equal | 1.0 / 1.0 / 1.0 | 620 | **100%** | 66.0 | 19.6 | 123.9 | 0.659 | 0.861 | 0.665 | 0.659 |
| self_fair_equal | 0.5 / 0.0 / 0.5 | 820 | **100%** | **67.7** | 19.7 | **131.6** | **0.677** | **0.910** | **0.760** | **0.677** |

## Key Findings

### Training improves negotiation quality across all metrics

All trained models outperform the base Qwen3-14B-Instruct model. Even the self_only model (trained purely on self-interest) shows modest gains, but the biggest improvements come from incorporating cooperative reward components.

### Agreement rate jumps with cooperative objectives

The base model and self_only model both fail to reach agreement in 30% of negotiations. Adding a fairness objective (fair_only) reduces failures to 20%, and both multi-objective models (all_equal, self_fair_equal) achieve 100% agreement rate. This suggests that cooperative reward signals teach the model to be more flexible and solution-oriented.

### self_fair_equal achieves the best overall outcomes (RQ2)

The model trained with lambda_self=0.5, lambda_welfare=0.0, lambda_fair=0.5 produces the highest scores across all metrics:

- **U_A = 67.7** (vs 44.1 baseline, +54%)
- **Social welfare = 131.6** (vs 88.2 baseline, +49%)
- **Ratio nash = 0.760** (vs 0.499 baseline, +52%)

This suggests that a weighted combination favoring self-interest while including welfare and fairness signals produces better negotiators than equal weighting (all_equal) or pure objectives (self_only, fair_only).

### Cooperative training reduces variance

The standard deviation of U_A drops from 33-36 (base/self_only) to ~20 (all_equal/self_fair_equal), indicating more consistent negotiation quality. The trained models don't just negotiate better on average -- they negotiate more reliably.

### Self-interest alone is insufficient

The self_only model shows the smallest improvement over baseline (+14% U_A, no change in agreement rate). Pure self-interest optimization doesn't teach the model to find mutually beneficial outcomes, which are necessary for reaching agreements in the first place.

## Eval Configurations

The 10 eval games cover a range of negotiation scenarios and archetypes:

1. **Single distributive** (rental agreement, 1 issue) -- x2 roles
2. **Two distributive issues** (loan agreement) -- x2 roles
3. **Distributive combo** (merger) -- x2 roles
4. **Compatible + distributive** (joint venture) -- x2 roles
5. **Integrative + compatible** (employment contract) -- x2 roles

## Caveats

- **Small sample size:** Only 10 games per model. Results should be validated with 50+ games for tighter confidence intervals.
- **Selfplay only:** The opponent is the frozen base model. Results against adversarial or OpenAI opponents may differ.
- **Single checkpoint per model:** Each model is evaluated at its final checkpoint only. Training dynamics (improvement over steps) are not captured here.
- **R_coop computed with lambda_self=1.0 only:** The ratio_rcoop column uses lambda_self=1.0, lambda_welfare=0.0, lambda_fair=0.0 for all models. To compare models on their own reward function, rerun with matching lambda values.
