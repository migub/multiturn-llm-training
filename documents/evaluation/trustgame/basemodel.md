# Trust Game Evaluation: Base Model (No LoRA)

**Date:** 2026-04-03  
**Model:** OpenPipe/Qwen3-14B-Instruct (4-bit quantized, no LoRA)  
**Games per role:** 50  
**Temperature:** 1.0  
**Seed:** 42  
**GPU:** RTX 5090 (CUDA 12.8)

---

## Results

|  | Trained as Investor | Trained as Trustee |
|--|--------------------|--------------------|
| Valid games | 50/50 (0% parse failure) | 50/50 (0% parse failure) |
| Avg send | 5.00 / 10 (std: 0.00) | 5.00 / 10 (std: 0.00) |
| Avg return | 10.92 (std: 1.01) | 10.80 (std: 0.99) |
| Return ratio | 72.8% | 72.0% |
| Investor payoff | 15.92 (std: 1.01) | 15.80 (std: 0.99) |
| Trustee payoff | 4.08 (std: 1.01) | 4.20 (std: 0.99) |
| Social welfare | 20.00 | 20.00 |
| Nash product | 64.0 | 65.4 |

**Reference points:** Nash equilibrium (10, 0) SW=10 NP=0 | Pareto optimum (15, 15) SW=30 NP=225

---

## Analysis

### Investor behavior: always sends 5 (zero variance)

The base model sends exactly 5 in every single game. This is the "safe middle ground" — half of the 10-point endowment. At temperature 0.6, the model is extremely confident in this choice. It's not copying the prompt example (which uses 6); it's choosing 5 as the intuitively fair/safe amount.

### Trustee behavior: over-returns (~72% of pool)

The Trustee receives 15 (5x3) and returns 10-12, keeping only 3-5 for itself. This is a 72% return ratio — far more generous than needed. A "fair" split would be returning ~50% (7-8 points), giving both players roughly equal payoffs.

This over-generosity is likely an artifact of RLHF training — the base model has been trained to be helpful and agreeable, which manifests as excessive reciprocation in the Trust Game.

### Payoff asymmetry

The Investor does extremely well (payoff ~16, even better than the Pareto optimum of 15) while the Trustee does poorly (payoff ~4). This is because the Trustee gives back too much. The base model's default behavior benefits the Investor at the Trustee's expense.

### Both roles produce identical results

Whether the trained model plays Investor or Trustee doesn't matter here — the base model plays the same way regardless. Both roles show send=5, return~11, because the base model (without LoRA) is the same in both positions. The small differences (10.92 vs 10.80 return) are just sampling noise.

---

## Implications for Trained Model Comparison

This baseline establishes what the untrained Qwen3-14B does by default:
- **Cautious investor** — sends exactly half, no risk-taking
- **Over-generous trustee** — returns too much, hurting own payoff
- **Social welfare = 20** — halfway between Nash (10) and Pareto (30)
- **Nash product = 65** — low due to extreme payoff asymmetry (16 vs 4)

Expected changes with trained models:
- **Self-only (lambda_fair=0):** Trustee should return less (keep more for self). Investor might send less.
- **Cooperative (lambda_fair>0):** Investor should send more (7-10). Trustee should return a fair share (~50%), pushing toward balanced payoffs and higher Nash product.
- **Key metric to watch:** Nash product. Base model = 65, Pareto = 225. Cooperative training should push this significantly higher by creating more balanced outcomes.
