# Trust Game Evaluation: Results Comparison

**Date:** 2026-04-03  
**Models compared:** Base model (no LoRA), Self-only (lambda_self=1.0), Fair-only (lambda_fair=1.0)  
**Games per role:** 50  
**Temperature:** 1.0  
**Opponent:** Frozen base model (Qwen3-14B-Instruct, no LoRA)  
**GPU:** RTX 5090 (CUDA 12.8)

---

## Results: Trained as Investor

The trained model plays Investor (sends points), the frozen base model plays Trustee (returns points).

| Metric | Base Model | Self-Only | Fair-Only | Pareto Optimum |
|--------|-----------|-----------|-----------|----------------|
| Avg send | 5.00 (std: 0.00) | 5.38 (std: 0.49) | **5.48** (std: 0.74) | 10 |
| Avg return | 11.02 | 10.42 | 10.56 | 15 |
| Return ratio | 73.5% | 65.7% | 65.8% | 50% |
| Investor payoff | 16.02 | 15.04 | 15.08 | 15 |
| Trustee payoff | 3.98 | 5.72 | 5.88 | 15 |
| Social welfare | 20.00 | 20.76 | **20.96** | 30 |
| Nash product | 62.5 | 80.8 | **83.1** | 225 |
| Parse failure | 0% | 0% | 0% | — |

## Results: Trained as Trustee

The frozen base model plays Investor (sends points), the trained model plays Trustee (returns points).

| Metric | Base Model | Self-Only | Fair-Only | Pareto Optimum |
|--------|-----------|-----------|-----------|----------------|
| Avg send | 5.00 | 5.00 | 5.00 | 10 |
| Avg return | 10.86 | 11.16 | **11.82** | 15 |
| Return ratio | 72.4% | 74.4% | **78.8%** | 50% |
| Investor payoff | 15.86 | 16.16 | **16.82** | 15 |
| Trustee payoff | 4.14 | 3.84 | **3.18** | 15 |
| Social welfare | 20.00 | 20.00 | 20.00 | 30 |
| Nash product | 64.4 | 59.3 | **47.0** | 225 |
| Parse failure | 0% | 0% | 0% | — |

---

## Analysis

### As Investor: small but positive transfer

Both trained models send slightly more than the base model (5.4-5.5 vs 5.0), breaking out of the zero-variance "always send half" pattern. Fair-only sends marginally more than self-only and achieves the highest Nash product (83.1 vs 62.5 baseline).

This demonstrates a measurable, if modest, **transfer of cooperative behavior from negotiation to the Trust Game**. The fair-only model learned to take slightly more risk (trust) as Investor, even though it was never trained on this game.

The improvement is small because the base model opponent (as Trustee) doesn't adjust — it returns ~65-73% regardless of how much is sent. There's no feedback loop where sending more gets rewarded with proportionally more returns.

### As Trustee: fair-only becomes too generous

This is the most interesting finding. The fair-only model returns *more* than the base model (78.8% vs 72.4%), making the Investor better off (payoff 16.82) but itself worse off (payoff 3.18). The Nash product actually **drops** from 64.4 (base) to 47.0 (fair-only).

The fair-only model learned generosity in negotiation — where both parties can adjust and reciprocate — but in the Trust Game's Trustee role, generosity is one-directional. The opponent (base model Investor) always sends 5 regardless, so the Trustee's extra generosity is simply exploited.

Self-only as Trustee is closer to the base model, returning slightly more (74.4% vs 72.4%) — a negligible difference.

### Why social welfare is locked at 20

Social welfare = 10 + 2 x sent. Since the base model Investor always sends exactly 5, social welfare is always 20 when the trained model plays Trustee. The Trustee can only redistribute the 20 points, not increase the total. Only the Investor role can influence social welfare, and even there the effect is small (20.76-20.96 vs 20.00).

### The opponent ceiling problem

The frozen base model is too deterministic — always sends 5, always returns ~72%. This creates a low ceiling for measurable effects:

- **As Investor:** The trained model can send more, but the base Trustee's return ratio drops slightly for higher sends, dampening the benefit
- **As Trustee:** The trained model can only redistribute a fixed pool of 15 points (5 x 3), so any "improvement" in one player's payoff comes at the other's expense

This is a structural limitation of evaluating against a frozen opponent, not a failure of the cooperative training.

---

## Key Takeaways

1. **Cooperative training transfers modestly to the Trust Game.** Fair-only as Investor shows higher trust (send 5.48 vs 5.00) and better Nash product (83.1 vs 62.5). Small effect, but in the right direction.

2. **Fair-only learns generosity, not strategic cooperation.** As Trustee, it gives back too much, hurting itself. In negotiation, generosity is reciprocated; in the Trust Game's Trustee role, it's exploited. The model learned a cooperative disposition but not game-specific strategy.

3. **Self-only and fair-only perform similarly as Investor.** The difference (5.38 vs 5.48 send, 80.8 vs 83.1 Nash product) is within noise. Both diverge from the base model in the same direction.

4. **The evaluation is limited by the deterministic opponent.** A more varied opponent (different temperatures, different models, or self-play) would likely reveal larger differences between training configurations.

---

## Possible Improvements

- **Self-play evaluation:** Both players are the trained model. Would show whether two cooperative agents can coordinate better than one cooperative + one frozen.
- **Note:** Temperature is already at 1.0, so the base model's send=5 lock is not a temperature issue — it's a strong prior from RLHF training.
- **Iterated Trust Game:** Multiple rounds where both players can learn to trust/defect based on history. Would test whether cooperative training produces robust reciprocity strategies.
- **Opponent variety:** Test against GPT-4o-mini or different checkpoints. McKee et al. (2020) showed diversity matters more than specific values — the same may apply to opponents.
