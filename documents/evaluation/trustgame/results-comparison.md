# Trust Game Evaluation: Full Results Comparison

**Date:** 2026-04-03  
**Models compared:** Base model (no LoRA), Self-only (lambda_self=1.0), Fair-only (lambda_fair=1.0), All-equal (lambda_self=0.33, lambda_welfare=0.33, lambda_fair=0.33)  
**Games per role:** 50  
**Temperature:** 1.0  
**Base model:** OpenPipe/Qwen3-14B-Instruct (4-bit quantized)  
**GPU:** RTX 5090 (CUDA 12.8)

---

## 1. vs_base: Trained Model vs Frozen Base Model

### As Investor (trained model sends, base model returns)

| Metric | Base Model | All-Equal | Self-Only | Fair-Only | Pareto |
|--------|-----------|-----------|-----------|-----------|--------|
| Avg send | 5.00 (std: 0.00) | 5.04 (std: 0.20) | 5.38 (std: 0.49) | **5.48** (std: 0.74) | 10 |
| Avg return | 11.02 | 10.90 | 10.42 | 10.56 | 15 |
| Return ratio | 73.5% | 72.3% | 65.7% | 65.8% | 50% |
| Investor payoff | 16.02 | 15.86 | 15.04 | 15.08 | 15 |
| Trustee payoff | 3.98 | 4.22 | 5.72 | 5.88 | 15 |
| Social welfare | 20.00 | 20.08 | 20.76 | **20.96** | 30 |
| Nash product | 62.5 | 65.1 | 80.8 | **83.1** | 225 |

### As Trustee (base model sends, trained model returns)

| Metric | Base Model | All-Equal | Self-Only | Fair-Only | Pareto |
|--------|-----------|-----------|-----------|-----------|--------|
| Avg send | 5.00 | 5.00 | 5.00 | 5.00 | 10 |
| Avg return | 10.86 | 11.36 | 11.16 | **11.82** | 15 |
| Return ratio | 72.4% | 75.7% | 74.4% | **78.8%** | 50% |
| Investor payoff | 15.86 | 16.36 | 16.16 | **16.82** | 15 |
| Trustee payoff | 4.14 | 3.64 | 3.84 | **3.18** | 15 |
| Social welfare | 20.00 | 20.00 | 20.00 | 20.00 | 30 |
| Nash product | 64.4 | 58.7 | 59.3 | **47.0** | 225 |

Parse failure rate: 0% across all models and roles.

---

## 2. vs_self: Self-Play (Trained vs Trained)

| Metric | Base Model | All-Equal | Self-Only | Fair-Only | Pareto |
|--------|-----------|-----------|-----------|-----------|--------|
| Avg send | 5.00 (std: 0.00) | 5.04 (std: 0.20) | 5.38 (std: 0.49) | **5.46** (std: 0.73) | 10 |
| Avg return | 11.02 | 11.16 | 10.70 | 10.98 | 15 |
| Return ratio | 73.5% | 74.0% | 67.5% | 68.4% | 50% |
| Investor payoff | 16.02 | 16.12 | 15.32 | 15.52 | 15 |
| Trustee payoff | 3.98 | 3.96 | 5.44 | 5.40 | 15 |
| Social welfare | 20.00 | 20.08 | **20.76** | 20.92 | 30 |
| Nash product | 62.5 | 62.3 | **76.2** | 74.9 | 225 |

---

## 3. Analysis

### All models are stuck near send=5

The most striking result is what didn't change. Across all models and both evaluation modes, the Investor sends between 5.00 and 5.48 out of 10. The base model's RLHF prior ("send half") is extremely strong, and LoRA fine-tuning on negotiation barely shifts it.

Even at temperature 1.0, the base model has zero variance on send. The trained models break this slightly (std 0.2-0.7), but the mean barely moves.

### Self-play did not unlock different behavior

Self-play results are nearly identical to vs_base as Investor. This makes sense — the Investor acts first without seeing the opponent, so it doesn't matter who the Trustee is. The hope was that a cooperative Trustee would somehow feed back into higher trust, but with a 2-turn game there's no feedback loop.

### Ranking by Nash product (the cooperative metric)

**vs_base as Investor** (best view of training effect):
1. Fair-only: 83.1 (+33% over base)
2. Self-only: 80.8 (+29%)
3. All-equal: 65.1 (+4%)
4. Base model: 62.5

**vs_self** (self-play):
1. Self-only: 76.2 (+22%)
2. Fair-only: 74.9 (+20%)
3. All-equal: 62.3 (~0%)
4. Base model: 62.5

Fair-only and self-only both improve over the base, but the difference between them is small and within noise. All-equal shows essentially no improvement — consistent with it being the weakest training configuration in the negotiation results.

### The Trustee problem: generosity without strategy

All trained models return 67-79% of the pool as Trustee — far more than the optimal 50% for an equal split. The fair-only model is the worst offender (78.8% vs_base), actually hurting its own Nash product by being too generous.

The models learned a cooperative *disposition* from negotiation training, but not cooperative *strategy*. In negotiation, being generous is reciprocated over multiple rounds. In the Trust Game's one-shot Trustee role, generosity is simply exploited.

### Social welfare is capped by send=5

Social welfare = 10 + 2 x sent. With all models sending ~5, welfare is locked at ~20. Only significantly higher trust (send 8-10) could push welfare toward the Pareto optimum of 30. No model comes close.

---

## 4. Key Findings for the Thesis

1. **Small but measurable transfer.** Cooperative negotiation training (fair-only, self-only) increases trust in the Trust Game by ~8-10% (send 5.4 vs 5.0). Nash product improves 20-33% over baseline. The effect is modest but consistent.

2. **Fair-only and self-only perform similarly.** Despite very different lambda configurations, both produce nearly identical Trust Game behavior. The negotiation-specific training signal doesn't translate into differentiated strategies in a structurally different game.

3. **All-equal training has no transfer effect.** All-equal (the weakest negotiation performer) shows no improvement in the Trust Game. Nash product is identical to the base model (62.3 vs 62.5). This reinforces that all-equal is an ineffective training configuration.

4. **Cooperative disposition, not cooperative strategy.** The trained models are slightly more trusting (Investor) and slightly more generous (Trustee), but they don't exhibit game-theoretically optimal behavior. A strategically cooperative Trustee would return ~50%, not 70-80%.

5. **The Trust Game is a hard out-of-domain test.** The modest transfer is not surprising — the models were trained on multi-issue negotiation with payoff tables, not on game-theoretic reasoning with explicit multipliers. The structural differences (2-turn vs multi-round, numerical decisions vs text agreements, asymmetric roles vs symmetric negotiators) make this a genuinely out-of-domain evaluation.

---

## 5. Limitations

- **Frozen base model opponent always sends 5.** This caps social welfare and limits the space for differentiation. The Trustee can only redistribute, not create value.
- **Self-play doesn't help.** Because the Trust Game is 2-turn sequential (Investor decides before seeing the Trustee), there's no feedback loop. The Investor's decision is independent of the opponent.
- **N=50 games.** With small effect sizes, statistical significance is questionable. The differences between self-only and fair-only (83.1 vs 80.8 Nash product) are within one standard deviation.
- **Single checkpoint per model.** Results may vary across training steps. Evaluating multiple checkpoints per run would show whether the effect is stable.
