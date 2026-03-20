# How Games Are Generated for Training

## Overview

Each training sample is a negotiation game consisting of 1 or 2 issues from the same scenario. The `create_dataset()` method in `env.py` builds all possible game configurations, permutes them, and cycles through them during training.

## Step 1: Define Scenarios and Issues

Each scenario (e.g., rental agreement, joint venture) has a set of issues:

| Scenario | Issues | Count |
|----------|--------|-------|
| Rental Agreement | rent, deposit, duration, duration-distributive | 4 |
| Loan Agreement | amount, duration, fees, rate | 4 |
| Merger | benefits, ownership | 2 |
| Joint Venture | rd-budget, revenue-split, data-sharing, decision-authority | 4 |
| Employment Contract | salary, remote-work, training-budget, equity, project-scope | 5 |

## Step 2: Generate All Combinations

For each scenario, the code creates:

1. **Single-issue games** — one game per issue (e.g., "rent only")
2. **Two-issue games** — every pair via `itertools.combinations(issues, 2)` (e.g., "rent + deposit")

Example for rental (4 issues):
- 4 single-issue games
- C(4,2) = 6 two-issue games (minus 1 excluded combo: duration + duration-distributive)
- Total: 9 game configs

Across all 5 scenarios this produces ~48 game configs (~19 single-issue + ~29 two-issue).

## Step 3: Assign Issue Weights

### Single-issue games
Weights are always `[[1], [1]]` — one issue, full importance for both sides.

### Two-issue games
Each side gets weights drawn randomly from `[90, 50, 10]`:

```python
iw_1 = random.choice([90, 50, 10])   # Side A's weight on issue 1
iw_2 = 100 - iw_1                     # Side A's weight on issue 2
# Same independently for Side B
```

So a side with weights `[90, 10]` cares 90% about issue 1 and 10% about issue 2.

## Why Issue Weights Matter

Issue weights are the mechanism that creates **integrative negotiation potential** — the ability for both sides to gain by trading concessions across issues.

### Example: Rent + Deposit

| | Side A (Landlord) | Side B (Tenant) |
|---|---|---|
| Rent weight | 90 (cares a lot) | 10 (doesn't care much) |
| Deposit weight | 10 (doesn't care much) | 90 (cares a lot) |

The payoffs for each issue are multiplied by these weights after normalization (`reweigh_issues()` in `games.py`). So:

- **Landlord** gets ~90% of total payoff from rent, ~10% from deposit
- **Tenant** gets ~10% of total payoff from rent, ~90% from deposit

The smart cooperative strategy is **logrolling**: the landlord gets a high rent (their priority), the tenant gets a low deposit (their priority). Both sides achieve ~90% of their maximum payoff. This is Pareto-efficient — neither side can improve without hurting the other.

### Contrast: Equal Weights

| | Side A | Side B |
|---|---|---|
| Rent weight | 50 | 50 |
| Deposit weight | 50 | 50 |

Both issues matter equally to both sides. There's no trade-off opportunity — what's good for one side on any issue is bad for the other. This is purely competitive.

## How Weights Determine Game Archetypes

The `get_game_type()` method in `games.py` classifies games based on:

1. **Do the weights differ between sides?** → Integrative (trade-offs possible)
2. **Are any issues compatible?** → Compatible (both prefer same outcome)

| Weights differ? | Has compatible issue? | Archetype |
|---|---|---|
| No | No | Non-integrative distributive (purely competitive) |
| No | Yes | Non-integrative compatible |
| Yes | No | Integrative distributive (trade-offs, no alignment) |
| Yes | Yes | Integrative compatible (trade-offs + alignment) |

The random draw from `[90, 50, 10]` creates a distribution of archetypes:
- When both sides draw `[50, 50]` → non-integrative (same priorities, no trade potential)
- When sides draw different weights like `[90, 10]` vs `[10, 90]` → integrative (strong trade potential)
- When sides draw similar weights like `[90, 10]` vs `[90, 10]` → non-integrative (same priorities)

## Step 4: Build Training Samples

The game configs are randomly permuted and cycled through:

```python
game_configs = np.random.permutation(game_configs)
for i in range(size // 2):
    game_config = game_configs[i % len(game_configs)]
```

Each game config produces **2 training samples** (one per role: agent plays Side A or Side B). So `train_size=1000` creates 500 game instances × 2 roles = 1000 samples.

## Issue Type Recap

Each individual issue has a fixed type that determines its payoff structure:

| Type | Payoff Pattern | Example |
|---|---|---|
| **Distributive** | A goes up, B goes down (zero-sum) | Rent: landlord wants high, tenant wants low |
| **Compatible** | Both go up together | R&D budget: both want more investment |
| **Integrative** | Both go in opposite directions but with different magnitudes | Remote work: company strongly prefers 0, employee strongly prefers 5 |

The issue type is inherent to the issue. The issue WEIGHTS (assigned per game) determine whether a 2-issue game has integrative potential through logrolling.
