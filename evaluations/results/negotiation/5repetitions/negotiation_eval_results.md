# Negotiation Evaluation Results (Multi-Game Selfplay, 5 Repetitions)

**Date:** 2026-04-08
**Eval script:** `evaluations/run_negotiation_eval.py`
**Setup:** 50 games per model (10 configs x 5 repetitions), selfplay with frozen base opponent, GPT-4o-mini judge, temperature=1.0, seed=42, max 5 rounds per negotiation.

## Results

| Model | Lambda (s/w/f) | Steps | Agreement | U_A mean | U_A std | Social Welfare | Ratio Self | Ratio Welfare | Ratio Nash | Ratio R_coop | Agreed R_coop |
|-------|---------------|-------|-----------|----------|---------|----------------|------------|---------------|------------|--------------|---------------|
| base_model | - | 0 | 76% | 50.1 | 31.8 | 99.7 | 0.501 | 0.655 | 0.541 | 0.501 | 0.659 |
| self_only | 1.0 / 0.0 / 0.0 | 560 | 80% | 55.1 | 33.2 | 104.8 | 0.551 | 0.696 | 0.564 | 0.551 | 0.689 |
| fair_only | 0.0 / 0.0 / 1.0 | 560 | **92%** | 61.6 | 23.7 | **121.8** | 0.616 | **0.834** | **0.723** | 0.616 | 0.670 |
| all_equal | 1.0 / 1.0 / 1.0 | 620 | **94%** | **65.0** | 22.7 | 121.9 | **0.650** | 0.827 | 0.663 | **0.650** | **0.691** |
| self_fair_equal | 0.5 / 0.0 / 0.5 | 820 | 90% | 62.2 | 26.8 | 120.7 | 0.622 | 0.805 | 0.679 | 0.622 | 0.691 |

## Key Findings

### All trained models improve over baseline

Every GRPO-trained model outperforms the base Qwen3-14B-Instruct across all metrics. The improvements are consistent across 50 games per model, confirming that the training signal is meaningful.

### Cooperative reward objectives drive the biggest gains (RQ2)

Self-only training (lambda_self=1.0) provides only modest improvements over baseline (+10% U_A, +4% agreement rate). Adding cooperative objectives yields much larger gains:

- **fair_only** (lambda_fair=1.0): +23% U_A, +16% agreement rate, best ratio_nash (0.723) and ratio_welfare (0.834)
- **all_equal** (all lambdas=1.0): +30% U_A, +18% agreement rate, highest absolute payoff (65.0)
- **self_fair_equal** (lambda_self=0.5, lambda_fair=0.5): +24% U_A, +14% agreement rate

### all_equal achieves the highest agent payoff and agreement rate

The model trained with equal weight on all three objectives (self-interest, welfare, fairness) reaches the best U_A (65.0) and agreement rate (94%). Balancing multiple objectives appears to produce the most well-rounded negotiator.

### fair_only produces the most efficient outcomes

Despite not optimizing for self-interest at all, fair_only achieves the highest ratio_nash (0.723) and ratio_welfare (0.834). Training on the Nash product (U_A x U_B) teaches the model to find mutually beneficial outcomes, which indirectly improves its own payoff.

### Cooperative training reduces variance

Standard deviation of U_A drops from 31.8-33.2 (base/self_only) to 22.7-26.8 (cooperative models). The cooperative models negotiate more consistently.

### Self-interest alone is insufficient

self_only shows the smallest improvement over baseline. Without cooperative signals, the model doesn't learn to find agreements or optimize joint outcomes effectively. This directly addresses RQ2: reward functions that balance self-interest with collective welfare produce better negotiators than pure self-interest.

## Eval Configurations

10 game configurations covering different negotiation scenarios and archetypes:

| # | Scenario | Issues | Archetype |
|---|----------|--------|-----------|
| 1-2 | Rental agreement | rent | single-distributive |
| 3-4 | Loan agreement | amount, rate | integrative compatible |
| 5-6 | Merger | benefits, ownership | non-integrative compatible |
| 7-8 | Joint venture | R&D budget, revenue split | non-integrative compatible |
| 9-10 | Employment contract | remote work, training budget | non-integrative compatible |

Each configuration is played from both starting roles (agent first vs opponent first), giving 10 configs. Each config is repeated 5 times with different seeds (seed 42-46).

## Caveats

- **Selfplay only:** The opponent is the frozen base model. Results against adversarial or OpenAI opponents may differ (RQ4).
- **Single checkpoint per model:** Each model is evaluated at its final checkpoint only. Training dynamics are captured in wandb.
- **R_coop computed with lambda_self=1.0 only:** The ratio_rcoop column uses lambda_self=1.0, lambda_welfare=0.0, lambda_fair=0.0 for all models. To compare models on their own reward function, rerun with matching lambda values.
- **In-domain evaluation:** All scenarios were seen during training. For generalization, run with `--game-type out-of-domain`.
