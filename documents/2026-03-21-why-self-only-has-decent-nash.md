# Why Self-Only Has Decent Nash Ratios — 2026-03-21

## Observation

At ~400 steps, the self-only run has `ratio_nash_mean ≈ 0.584`, which is only slightly below fair-only's `0.656`. This seems surprising since self-only doesn't optimize for Nash product at all.

## Comparison (average of last 5 logged entries)

| Metric | Self-Only | Fair-Only | Better |
|---|---|---|---|
| ratio_nash | 0.584 | **0.656** | Fair |
| ratio_self | **0.595** | 0.550 | Self |
| ratio_welfare | 0.720 | **0.798** | Fair |
| agreement_rate | 0.875 | **0.945** | Fair |
| agreed/ratio_self | **0.66-0.77** | 0.50-0.77 | Self |
| agreed/ratio_welfare | 0.68-0.94 | 0.70-0.91 | Similar |

Fair-only is better on nash (+0.07), welfare (+0.08), and agreement rate (+0.07). Self-only is better on self (+0.05). The differences are real but moderate.

## Why the Gap Isn't Bigger

### 1. Self-interest indirectly optimizes Nash

To get high U_A, the agent needs to reach an agreement (otherwise reward=0). And reaching an agreement means the opponent gets something too. So self-interest implicitly pushes for:
- High agreement rate (needed for any reward)
- Reasonable U_B (opponent needs to accept the deal)
- Therefore decent U_A × U_B

A purely selfish agent that never compromises gets reward=0 — GRPO punishes this. So "good selfishness" naturally produces okay Nash products.

### 2. Failed negotiations drag down self-only's metrics

Self-only has 87.5% agreement rate vs 94.5% for fair-only. The 12.5% failures contribute 0 to all ratio metrics. If you only look at successful negotiations (`agreed/*` metrics), the quality gap is even smaller.

### 3. Frozen opponent creates a ceiling

Both agents negotiate against the same frozen local model. There's a maximum Nash product achievable against this fixed opponent. Both runs are approaching this ceiling from different angles — self-only by maximizing U_A, fair-only by balancing U_A and U_B.

### 4. Runs may not be fully converged

Fair-only has higher KL divergence (0.22-0.62 vs 0.07-0.13), suggesting it's still learning more aggressively. The gap could widen with more training.

## Key Takeaway for Thesis

The **strongest differentiator is agreement rate**, not Nash ratio. Fair-only almost always reaches a deal (0.95+), while self-only still fails 12.5% of the time. This is the most compelling argument for cooperative reward functions:

> Nash product reward doesn't just improve fairness — it fundamentally improves the agent's ability to reach agreements, because any failure results in 0 × anything = 0 reward.

For the Nash ratio itself, the difference exists but is moderate. This could mean:
- Nash product provides diminishing returns over self-interest alone (finding for RQ2)
- The combined run (self+nash) might capture the best of both: high U_A from self-interest + high agreement rate from Nash
- The all-equal run will test whether adding welfare on top changes anything
