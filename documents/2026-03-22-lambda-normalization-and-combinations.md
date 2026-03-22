# Lambda Normalization and Sensible Combinations

**Date:** 2026-03-22
**Context:** Choosing lambda values for the cooperative reward function R_coop

---

## The Problem: Scale Mismatch

The cooperative reward combines three objectives:

```
R_coop = lambda_self * U_A + lambda_welfare * (U_A + U_B) + lambda_fair * (U_A * U_B) / 100
```

These three components have **very different scales** depending on the game:

| Component       | Formula           | Typical Range | Example (2-issue compatible game) |
|-----------------|-------------------|---------------|-----------------------------------|
| Self (U_A)      | U_A               | 0–100         | max = 100                         |
| Welfare         | U_A + U_B         | 0–200         | max = 200 (both get 100)          |
| Nash/Fairness   | U_A * U_B / 100   | 0–100         | max = 100 (100*100/100)           |

With equal lambdas (1.0, 1.0, 1.0), the welfare term dominates simply because its raw values are larger — not because we intended it to matter more.

**Why this matters for GRPO:** The training signal comes from the advantage `A_i = (R_i - mean(R)) / std(R)` within a group. If one component's variance dominates, only that component drives the gradient. The model effectively ignores the other objectives.

### Additional scale issues

- **Across game types:** A distributive game might have max U_A = 80, while a compatible game has max U_A = 100. Without normalization, the model gets stronger gradient signal from compatible games.
- **Welfare is constant in pure distributive games:** When U_A + U_B = constant (zero-sum), welfare has zero variance and provides no learning signal at all. This was already noted, but with unnormalized rewards the welfare lambda still shifts the reward mean upward, wasting "budget."
- **Nash product scale depends on both players:** U_A * U_B can range from 0 to 10,000 (before /100), making its contribution highly variable across games.

---

## The Fix: Normalize Before Weighting

We now compute R_coop using **game-normalized ratios** instead of raw values:

```
R_coop = lambda_self  * (U_A / max_U_A)
       + lambda_welfare * ((U_A + U_B) / max_welfare)
       + lambda_fair    * ((U_A * U_B) / max_nash)
```

Each component is now in **[0, 1]** regardless of game type. The `compute_max_metrics()` function already brute-forces the per-game maximum of each component (at most 121 combinations for 2-issue games), so this required minimal code change.

### What this achieves

1. **Lambdas are directly interpretable** as relative importance weights
2. **Fair comparison across game types** — a compatible game and a distributive game contribute equally to the gradient
3. **No component dominates by accident** — the model receives balanced signal from all three objectives
4. **max_r_coop is also normalized** — computed in a second pass using the same normalized formula, so `ratio_rcoop` remains consistent

---

## Sensible Lambda Combinations

Since all components are now in [0, 1], the lambdas don't need to sum to 1 (GRPO normalizes advantages anyway), but summing to 1 makes them easy to interpret as "percentage of importance."

### Recommended configurations

| Name | lambda_self | lambda_welfare | lambda_fair | Total | Use Case |
|------|-------------|----------------|-------------|-------|----------|
| **Self-only (baseline)** | 1.0 | 0.0 | 0.0 | 1.0 | Luca's original setup. Pure self-interest. Comparison baseline. |
| **Balanced** | 0.4 | 0.3 | 0.3 | 1.0 | Good default. Self-interest slightly prioritized, both cooperative signals included. |
| **Nash-heavy** | 0.3 | 0.2 | 0.5 | 1.0 | Nash product inherently rewards efficiency AND fairness. Best single cooperative metric. |
| **Welfare-heavy** | 0.3 | 0.5 | 0.2 | 1.0 | Maximizes total pie. Risk: can learn unfair splits that still score well. |
| **Equal** | 1/3 | 1/3 | 1/3 | 1.0 | Neutral starting point for ablation. |
| **Cooperative-dominant** | 0.2 | 0.4 | 0.4 | 1.0 | Strong cooperative signal. Self-interest only as stabilizer. |
| **Fair-only** | 0.0 | 0.0 | 1.0 | 1.0 | Pure Nash product optimization. Extreme, but useful for ablation. |

### Rationale for each component

**Self-interest (lambda_self):**
- Keeps the agent from sacrificing itself entirely for the opponent
- Stabilizes training — without it, the agent might learn degenerate strategies where it gives everything away (which scores well on welfare/nash but isn't realistic)
- Recommendation: always keep >= 0.2

**Welfare (lambda_welfare):**
- Encourages "growing the pie" — finding outcomes where both parties benefit more
- Strong signal in compatible and integrative games where win-win outcomes exist
- **Weakness:** Zero signal in pure distributive games (U_A + U_B = constant). In the current game set, ~7% of games are pure compatible (welfare varies), ~69% are mixtures, ~24% are integrative-distributive. So welfare does provide signal in most games, but less than the other components.
- Recommendation: 0.2–0.4

**Fairness/Nash (lambda_fair):**
- Nash product (U_A * U_B) is arguably the most theoretically grounded cooperative metric:
  - It's the **Nash bargaining solution** — the standard game theory concept for fair negotiation outcomes
  - It's **zero if either party gets zero** — strongly punishes exploitation
  - It's **maximized when the pie is both large AND fairly split** — combines efficiency and fairness in one metric
  - It provides signal even in pure distributive games (unlike welfare)
- Recommendation: 0.3–0.5

---

## Ablation Strategy for RQ2

RQ2 asks: *How should reward functions balance self-interest vs. collective welfare?*

### Phase 1: Prove normalization matters

| Run | lambda_self | lambda_welfare | lambda_fair | Note |
|-----|-------------|----------------|-------------|------|
| A   | 1.0 | 0.5 | 0.3 | **Unnormalized** (old code) |
| B   | 1.0 | 0.5 | 0.3 | **Normalized** (new code) |

Compare: does the normalized version actually improve cooperative metrics while maintaining self-interest? If welfare dominates in run A but all three contribute in run B, normalization is validated.

### Phase 2: Vary one lambda at a time

Hold the other two equal and sweep one:

```
lambda_self  in {0.1, 0.3, 0.5, 0.7}  with lambda_w = lambda_f = (1 - lambda_s) / 2
lambda_fair  in {0.1, 0.3, 0.5, 0.7}  with lambda_s = lambda_w = (1 - lambda_f) / 2
```

This traces a path from "mostly cooperative" to "mostly selfish" (or "mostly fair") and shows the Pareto trade-off between objectives.

### Phase 3: Key comparisons

| Run | lambda_self | lambda_welfare | lambda_fair | Question answered |
|-----|-------------|----------------|-------------|-------------------|
| Self-only | 1.0 | 0.0 | 0.0 | Baseline: how cooperative is pure self-interest? |
| Nash-only | 0.0 | 0.0 | 1.0 | Can Nash product alone teach cooperation? |
| Welfare vs Nash | 0.3 | 0.7 | 0.0 vs 0.3 | 0.0 | 0.7 | Does welfare or Nash produce better cooperative agents? |
| Balanced | 0.4 | 0.3 | 0.3 | Best combined signal? |

### Phase 4: Game type interaction (if time permits)

Run the best lambda config on:
- `multi-game` (all scenarios) — does it generalize?
- `cooperative-only` (JV + EC) — is cooperative signal stronger when all games have cooperative potential?

---

## Intuition: Why Nash Product is Special

Consider a simple distributive game where U_A + U_B = 100:

| U_A | U_B | Welfare | Nash Product |
|-----|-----|---------|-------------|
| 100 | 0   | 100     | 0           |
| 80  | 20  | 100     | 1600        |
| 60  | 40  | 100     | 2400        |
| 50  | 50  | 100     | **2500**    |
| 40  | 60  | 100     | 2400        |

Welfare is constant — it can't distinguish outcomes. Nash product peaks at the fair split. Now consider an integrative game where choices matter:

| Outcome | U_A | U_B | Welfare | Nash Product |
|---------|-----|-----|---------|-------------|
| Compete | 60  | 60  | 120     | 3600        |
| Trade   | 80  | 70  | **150** | **5600**    |
| Exploit | 90  | 30  | 120     | 2700        |

Nash product rewards the efficient trade AND penalizes exploitation — exactly the behavior we want. Welfare also rewards the trade but doesn't distinguish it from the exploit outcome as strongly.

This is why lambda_fair deserves significant weight in the cooperative reward.

---

## Implementation Details

The change was made in `envs/negotiation/env.py`:

1. **`compute_max_metrics()`** — Now uses two passes: first computes per-component maxima (max_U_A, max_social_welfare, max_nash_product), then computes max_r_coop using normalized components in a second loop.

2. **Reward computation in `get_rewards()`** — R_coop now uses `ratio_self`, `ratio_welfare`, `ratio_nash` (all in [0,1]) instead of raw U_A, welfare, nash_product_normalized. The old `/100` division for Nash product is replaced by proper game-specific normalization.

3. **Backward compatibility** — With `lambda_self=1.0, lambda_welfare=0.0, lambda_fair=0.0`, the reward is just `U_A / max_U_A` (i.e., ratio_self). This is a monotonic transformation of the old reward, so GRPO advantages and ranking are identical. Self-only runs are unaffected.
