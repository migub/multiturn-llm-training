# LA-GRPO Evaluation Results (20 repetitions)

Comparison of three LA-GRPO training objectives evaluated on the same held-out set, 20 repetitions per game configuration.

Source files:
- `lagrpo_self_only_v2_2000.json` (λ_self=1, λ_welfare=0, λ_fair=0, step 2000)
- `lagrpo_fair_only_1340.json` (λ_self=0, λ_welfare=0, λ_fair=1, step 1340)
- `lagrpo_all_equal_2000.json` (λ_self=1, λ_welfare=1, λ_fair=1, step 2000)

## Aggregate Metrics (all games, including no-agreement)

| Run | Agree | U_A | U_B | SW | r_self | r_welf | r_nash |
|---|---|---|---|---|---|---|---|
| self (2000) | **0.955** | **68.35** | 60.20 | **128.55** | **0.683** | **0.896** | 0.684 |
| fair (1340) | 0.880 | 61.70 | 60.00 | 121.70 | 0.617 | 0.832 | **0.693** |
| equal (2000) | 0.920 | 60.05 | **61.15** | 121.20 | 0.601 | 0.857 | 0.680 |

On aggregates, `self` leads on agreement rate, U_A, SW, and r_self. However, much of this advantage comes from closing more deals (non-agreements contribute 0 to the mean), which inflates every raw metric. Once we condition on agreement, the picture changes.

## Agreed-Only Aggregate Metrics

| Run | r_self | r_welf | r_nash | r_coop |
|---|---|---|---|---|
| self (2000) | **0.716** | 0.938 | 0.716 | 0.716 |
| fair (1340) | 0.701 | **0.945** | **0.787** | 0.787 |
| equal (2000) | 0.653 | 0.931 | 0.740 | **0.847** |

Per-deal quality: `fair` produces the highest Nash product and social welfare; `equal` maximises its own (blended) training objective; `self` is the most aggressive in extracting own-payoff.

## Per-Archetype Agreed-Only Metrics

### ratio_self (own-payoff captured, agreed only)

| Archetype | self | fair | equal | Winner |
|---|---|---|---|---|
| distributive | **0.560** | 0.544 | 0.476 | self |
| compatible | 0.885 | **0.891** | 0.853 | fair |
| integrative | **0.658** | 0.559 | 0.589 | self |

### ratio_welfare (social welfare, agreed only)

| Archetype | self | fair | equal | Winner |
|---|---|---|---|---|
| distributive | 1.000 | 1.000 | 1.000 | tie (SW conserved) |
| compatible | 0.885 | **0.891** | 0.853 | fair |
| integrative | 0.932 | **0.970** | 0.955 | fair |

### ratio_nash (Nash product — key cooperation metric, agreed only)

| Archetype | self | fair | equal | Winner |
|---|---|---|---|---|
| distributive | 0.633 | **0.680** | 0.676 | fair |
| compatible | 0.793 | **0.802** | 0.735 | fair |
| integrative | 0.715 | **0.920** | 0.872 | **fair** |

**Fair wins Nash on every archetype once we control for agreement rate.** This is the central cooperative-training result.

### U_A / U_B balance (agreed only)

| Archetype | self | fair | equal |
|---|---|---|---|
| distributive | 56.0 / 44.0 | 54.4 / 45.6 | **47.6 / 52.4** |
| compatible | 88.5 / 88.5 | 89.1 / 89.1 | 85.3 / 85.3 |
| integrative | 65.8 / 46.1 | 55.9 / **60.5** | 58.9 / 55.7 |

`self` splits in its own favour (56/44 distributive, 66/46 integrative). `fair` and `equal` produce more balanced outcomes; `equal` on distributive even concedes more than half to the opponent on average.

### Agreement counts (agreed games out of 80 / 80 / 40)

| Archetype | self | fair | equal |
|---|---|---|---|
| distributive | **73** | 59 | 72 |
| compatible | **80** | 78 | 75 |
| integrative | 38 | **39** | 37 |

`self` closes the most distributive deals, which drives the aggregate-level advantage on U_A and SW. `fair` closes fewer distributive deals but produces higher-quality agreements when it does.

## Interpretation for the Thesis

1. **Aggregate "self wins" is partly an artefact of agreement rate.** Raw U_A and SW are inflated by closing more deals, not by producing better deals.
2. **Fair training optimises Nash product on every archetype.** This is the cleanest evidence that explicit cooperative reward shapes per-deal outcome quality, especially on integrative games (0.920 vs 0.715 for self).
3. **Equal (blended) reward produces the most balanced U_A / U_B splits** — arguably the fairest behaviour — but does not dominate any single objective. The blended signal may be diluting learning gradients.
4. **Archetype matters.** No single reward shape dominates; the right λ depends on the game type. This is direct evidence for RQ2 (how should rewards balance self vs. collective welfare?).

## Caveats

- Step counts are not matched: `fair_only` ran to 1340, the other two to 2000. `fair_only` may still have been improving.
- `ratio_rcoop` is defined differently per run (depends on each run's own λ-weights) and is not cross-comparable. Use `ratio_nash`, `ratio_welfare`, and the agreed-only ratios for cross-method comparison.
- In pure distributive games, `ratio_welfare` is always 1.0 because U_A + U_B is conserved; it provides no signal there.
- Aggregate means include no-agreement (payoff 0) rows. The agreed-only block is the right view for comparing negotiation quality.

## Suggested Next Checks

- Pull wandb training curves to see whether `all_equal` plateaued early or was still improving at step 2000.
- Match step counts across all three LA-GRPO runs for a fair comparison.
- Re-score every run under each of the three reward shapes (self, fair, equal) to show each run wins its own objective — the clean ablation figure for the thesis.
