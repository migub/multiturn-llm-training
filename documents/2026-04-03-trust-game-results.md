# Trust Game Evaluation Results

**Date:** 2026-04-03  
**Purpose:** Compare cooperative training variants on the Trust Game (out-of-domain generalization).

---

## Setup

- **Base model:** OpenPipe/Qwen3-14B-Instruct (4-bit quantized)
- **Games per condition:** 50
- **Temperature:** 1.0
- **Seed:** 42
- **Parse failures:** 0% across all conditions

### Models Evaluated

| Model | Checkpoint | Lambda Config |
|-------|-----------|---------------|
| Base model | none (no LoRA) | n/a |
| Self-only | `grpo-multigame-self-only/checkpoint-560` | lambda_self=1.0, lambda_welfare=0, lambda_fair=0 |
| Fair-only | `grpo-multigame-fair-only/checkpoint-560` | lambda_self=0, lambda_welfare=0, lambda_fair=1.0 |
| All-equal | `grpo-multigame-all-equal/checkpoint-620` | lambda_self=1.0, lambda_welfare=1.0, lambda_fair=1.0 |

### Reference Points

| Outcome | Send | Return | Investor | Trustee | Social Welfare | Nash Product |
|---------|------|--------|----------|---------|----------------|--------------|
| Nash equilibrium (selfish) | 0 | 0 | 10 | 0 | 10 | 0 |
| Pareto optimum (cooperative) | 10 | 15 | 15 | 15 | 30 | 225 |

---

## Results: Selfplay (Trained vs Trained)

Both roles played by the same model with LoRA enabled.

| Metric | Base Model | Self-Only | Fair-Only | All-Equal |
|--------|-----------|-----------|-----------|-----------|
| Avg Send | 5.00 (+-0.00) | 5.38 (+-0.49) | 5.46 (+-0.73) | 5.04 (+-0.20) |
| Avg Return | 11.02 (+-1.15) | 10.70 (+-1.98) | 10.98 (+-2.58) | 11.16 (+-1.06) |
| Return Ratio | 73.5% | 67.5% | 68.4% | 74.0% |
| Investor Payoff | 16.02 | 15.32 | 15.52 | 16.12 |
| Trustee Payoff | 3.98 | 5.44 | 5.40 | 3.96 |
| **Social Welfare** | 20.00 | **20.76** | **20.92** | 20.08 |
| **Nash Product** | 62.5 | **76.2** | **74.9** | 62.3 |

---

## Results: Vs Base (Trained vs Frozen Base)

Trained model plays one role, frozen base model (LoRA disabled) plays the other.

### As Investor (trained sends, base returns)

| Metric | Base Model | Self-Only | Fair-Only | All-Equal |
|--------|-----------|-----------|-----------|-----------|
| Avg Send | 5.00 (+-0.00) | 5.38 (+-0.49) | 5.48 (+-0.74) | 5.04 (+-0.20) |
| Avg Return | 11.02 (+-1.15) | 10.42 (+-1.57) | 10.56 (+-1.70) | 10.90 (+-1.20) |
| Return Ratio | 73.5% | 65.7% | 65.8% | 72.3% |
| Investor Payoff | 16.02 | 15.04 | 15.08 | 15.86 |
| Trustee Payoff | 3.98 | 5.72 | 5.88 | 4.22 |
| **Social Welfare** | 20.00 | **20.76** | **20.96** | 20.08 |
| **Nash Product** | 62.5 | **80.8** | **83.1** | 65.1 |

### As Trustee (base sends, trained returns)

| Metric | Base Model | Self-Only | Fair-Only | All-Equal |
|--------|-----------|-----------|-----------|-----------|
| Avg Send | 5.00 | 5.00 | 5.00 | 5.00 |
| Avg Return | 10.86 (+-1.14) | 11.16 (+-1.68) | 11.82 (+-2.58) | 11.36 (+-0.94) |
| Return Ratio | 72.4% | 74.4% | 78.8% | 75.7% |
| Investor Payoff | 15.86 | 16.16 | 16.82 | 16.36 |
| Trustee Payoff | 4.14 | 3.84 | 3.18 | 3.64 |
| **Social Welfare** | 20.00 | 20.00 | 20.00 | 20.00 |
| **Nash Product** | 64.4 | 59.3 | 47.0 | 58.7 |

---

## Interpretation

### 1. Self-only and fair-only improve cooperation in selfplay

Both self-only and fair-only show meaningful improvements over the base model in selfplay:
- Social welfare increases by ~4-5% (20.0 -> 20.8-20.9)
- Nash product increases by ~20% (62.5 -> 74.9-76.2)
- Payoff distribution is more balanced (trustee gets ~5.4 vs ~4.0 for base)

This is notable because these models were **never trained on the Trust Game**. The cooperative behavior transfers from multi-issue negotiation to a structurally different game.

### 2. All-equal performs indistinguishably from base model

The all-equal model (lambda_self=lambda_welfare=lambda_fair=1.0) shows virtually no improvement over the base model across all conditions. This suggests that **mixing all three reward signals equally creates conflicting gradients** that cancel each other out, preventing effective learning.

This is an important negative result for **RQ2** (how to balance reward function components): focused reward signals (pure self-interest or pure fairness) produce clearer learning than a kitchen-sink approach.

### 3. Self-only discovers cooperation instrumentally

Counter to the initial hypothesis that self-only models would converge toward Nash equilibrium, the self-only model actually becomes *more* cooperative than baseline. In selfplay, it sends more (5.38 vs 5.00), achieves better social welfare, and produces more balanced outcomes.

**Why?** In multi-issue negotiation training, a self-interested agent learns that cooperative strategies (making concessions to reach agreement) yield higher personal payoffs than stonewalling. This instrumental cooperation transfers: in the Trust Game, a model trained to maximize its own payoff has learned that giving-to-get-back is a viable strategy.

### 4. As Trustee vs base: fair-only is too generous

When the fair-only model plays Trustee against the frozen base Investor (which always sends exactly 5), it returns 78.8% of the pool — far more than the base model's 72.4%. This over-generosity makes it exploitable:
- Investor payoff rises to 16.82 (best of all conditions)
- Trustee payoff drops to 3.18 (worst of all conditions)
- Nash product drops to 47.0 (worst of all conditions)

The fair-only model learned to prioritize balanced outcomes, but when paired with a rigid, non-reciprocating opponent, this altruism becomes self-sacrificing. This is relevant to **RQ4** (robustness against adversarial opponents).

### 5. Base model is extremely rigid

The base Qwen3-14B-Instruct always sends exactly 5 as Investor (std=0.00) across all conditions. Training breaks this rigidity — self-only and fair-only show variance in send amounts (std 0.49-0.73), suggesting they've learned to explore different strategies. All-equal remains almost as rigid as base (std=0.20).

---

## Summary Table (Key Metrics)

| Model | Selfplay SW | Selfplay NP | Vs-Base NP (Investor) | Vs-Base NP (Trustee) |
|-------|------------|------------|----------------------|---------------------|
| Base model | 20.00 | 62.5 | 62.5 | 64.4 |
| Self-only | **20.76** | **76.2** | **80.8** | 59.3 |
| Fair-only | **20.92** | 74.9 | **83.1** | 47.0 |
| All-equal | 20.08 | 62.3 | 65.1 | 58.7 |

---

## Implications for the Thesis

### Positive Findings

1. **Cooperative training generalizes out-of-domain** — Both self-only and fair-only improve Trust Game outcomes despite never being trained on it. This is evidence that RL-based negotiation training produces genuinely cooperative agents, not just negotiation-specific heuristics.

2. **Self-interest can lead to cooperation** — The self-only model discovering cooperative behavior instrumentally is a compelling narrative: you don't need to explicitly train for cooperation if the environment rewards it. This supports the broader argument that well-designed reward functions can shape pro-social behavior.

3. **Lambda ablation yields clear findings for RQ2** — The all-equal negative result + self-only/fair-only positive results provide concrete guidance: focused rewards outperform mixed rewards. Future work could explore intermediate weightings (e.g., lambda_self=0.7, lambda_fair=0.3).

### Limitations

1. **Small effect sizes** — Social welfare improvements are modest (~5%). The Trust Game may not be sensitive enough to capture the full difference between models.

2. **Ceiling effect from frozen opponent** — When playing vs base (which always sends 5), social welfare is structurally capped at 20 regardless of Trustee behavior. Selfplay is the better test for cooperation.

3. **N=50 per condition** — Standard deviations are large relative to effect sizes. Statistical significance tests (Mann-Whitney U) should be run before drawing strong conclusions.

4. **Single checkpoint per model** — Results reflect one point in training (step 560/620). Training dynamics (how metrics evolve over steps) may tell a richer story.

---

## Files

- **Script:** `evaluations/trust_game_eval.py`
- **Runner:** `evaluations/run_selfplay_evals.sh`
- **Results (vs_base):** `evaluations/results/trustgame/vs_base/{base_model,self_only,fair_only,all_equal}.json`
- **Results (selfplay):** `evaluations/results/trustgame/vs_self/{base_model,self_only,fair_only,all_equal}.json`
