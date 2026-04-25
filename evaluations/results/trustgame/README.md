# Trust Game — LA-GRPO Transfer Evaluation

Out-of-domain evaluation of the three LA-GRPO checkpoints on the one-shot Trust Game. The goal is to probe whether cooperative negotiation training transfers to an unrelated cooperation game (RQ5: capability preservation / transfer).

## What was run

- **Models:** `base_model` (Qwen3-14B-Instruct, no LoRA), `lagrpo_self_only` (ckpt-2000), `lagrpo_all_equal` (ckpt-2000), `lagrpo_fair_only` (ckpt-1340).
- **Modes:**
  - `vs_base/` — trained model as Investor vs. frozen base as Trustee, and vice versa.
  - `vs_self/` — trained model plays both roles (self-play, LoRA on for both turns).
- **Seed 42, temperature 1.0, max_new_tokens 200.**
- **Two scales:** 100 games/role (top-level dirs) and 300 games/role (`300games/` dir).
- Driver script: `evaluations/run_lagrpo_trust_game_eval.sh` (parametrized via `NUM_GAMES` and `RESULTS_TAG` env vars).

All runs finished with **0 parse failures** and **0 runtime errors**.

## Results (300 games/role)

### vs_base

| Model | Role | send | return | ret.ratio | I payoff | T payoff | SW | NP |
|---|---|---|---|---|---|---|---|---|
| base_model       | investor | 5.00 | 11.09 | 74.0% | 16.09 | 3.91 | 20.00 | 61.1 |
| base_model       | trustee  | 5.00 | 11.17 | 74.5% | 16.17 | 3.83 | 20.00 | 60.3 |
| lagrpo_self_only | investor | 5.22 | 10.78 | 69.7% | 15.56 | 4.87 | 20.43 | 71.3 |
| lagrpo_self_only | trustee  | 5.00 | 11.38 | 75.8% | 16.38 | 3.62 | 20.00 | 57.0 |
| lagrpo_all_equal | investor | 5.06 | 11.20 | 74.0% | 16.13 | 3.99 | 20.13 | 61.4 |
| lagrpo_all_equal | trustee  | 5.00 | 10.71 | 71.4% | 15.71 | 4.29 | 20.00 | 65.6 |
| lagrpo_fair_only | investor | 5.05 | 11.07 | 73.3% | 16.02 | 4.07 | 20.09 | 62.9 |
| lagrpo_fair_only | trustee  | 5.00 | 11.08 | 73.9% | 16.08 | 3.92 | 20.00 | 61.3 |

### vs_self (self-play)

| Model | send | return | ret.ratio | I payoff | T payoff | SW | NP |
|---|---|---|---|---|---|---|---|
| base_model       | 5.00 | 11.09 | 74.0% | 16.09 | 3.91 | 20.00 | 61.1 |
| lagrpo_self_only | 5.22 | 10.97 | 70.9% | 15.75 | 4.68 | 20.43 | 68.4 |
| lagrpo_all_equal | 5.06 | 10.83 | 71.6% | 15.77 | 4.36 | 20.13 | 65.7 |
| lagrpo_fair_only | 5.05 | 10.91 | 72.3% | 15.87 | 4.23 | 20.09 | 64.6 |

**Reference points:** Nash equilibrium (selfish) = (10, 0), SW=10, NP=0. Pareto optimum = (15, 15), SW=30, NP=225.

### Welch t-tests vs. base (self-play, n=300 each)

Bonferroni-corrected α = 0.0167 for three LA-GRPO comparisons.

| Model | Δ send | Δ return | Δ SW | Δ NP |
|---|---|---|---|---|
| lagrpo_self_only | **+0.22** *** | −0.13 n.s. | **+0.43** *** | **+7.27** *** |
| lagrpo_all_equal | **+0.06** *** | −0.26 (*)  | **+0.13** *** | **+4.66** *** |
| lagrpo_fair_only | **+0.05** *** | −0.18 n.s. | **+0.09** *** | +3.56 (*)    |

`***` p < 0.0167 after Bonferroni. `(*)` p < 0.05 but not after correction. n.s. = not significant.

Note that `SW = 10 + 2·send` in the Trust Game, so the SW shift is mechanically driven by the send shift — the two numbers are not independent evidence.

## Interpretation

1. **All three LA-GRPO models send slightly more than base**, with `lagrpo_self_only` showing the largest shift (+0.22). The effect is statistically significant but very small (~2% of scale).

2. **Ordering is inverted from theory.** If λ-weighted cooperative training were the driver, we would expect `lagrpo_all_equal` ≥ `lagrpo_fair_only` > `lagrpo_self_only`. Instead the self-only model moves the most. This suggests the shift is an artefact of GRPO fine-tuning in general (more variance in the policy, less deterministic output), not of the cooperative reward weights.

3. **Return ratio trends downward**, not upward. Trustees return slightly less of what they receive. This is the opposite of what "cooperative training transfers to other cooperation games" would predict.

4. **Base model is nearly deterministic** (send=5.00, std=0.00). LA-GRPO introduces variance (std 0.21–0.41), which is where the mean send shift comes from — not from a qualitatively different strategy.

## Caveats

- **Trust Game is one-shot** → the game-theoretic equilibrium for the Trustee is return = 0. That base returns 74% reflects Qwen's pretraining bias, not any trained cooperation. Small post-training shifts on top of this bias are hard to interpret.
- **No multi-issue structure** → the Trust Game has no payoff matrix, no compatible/integrative archetypes, no issue-level trade-offs. It probes a very different decision surface from what the models were trained on.
- **Temperature 1.0, single seed** → a lower-temperature sweep with multiple seeds would better distinguish strategy shifts from sampling noise.
- **Parameter ratio problem for `lagrpo_self_only`**: the `Δ send` comes almost entirely from cases where the model sends 6 instead of 5. This is a one-digit nudge, not a strategic change.

## Takeaways for the thesis

- **Use Trust Game as a capability-preservation / transfer probe (RQ5), not as primary evidence for cooperation gains (RQ1/RQ2).** The Trust Game results show the trained models remain coherent and usable at an unrelated negotiation-adjacent task — no catastrophic regression, full parse success.
- **The contrast with the in-domain negotiation evals is the interesting story.** Large shifts there + tiny shifts here → cooperative training is domain-specific, consistent with what fine-tuning on narrow tasks typically produces.
- **Do not argue that λ-cooperative training made the models "trust more"** based on these numbers alone; the effect size is too small and the ordering contradicts the mechanism.

## Files

```
evaluations/results/trustgame/
├── README.md                             ← this file
├── vs_base/*.json                        ← 100 games/role
├── vs_self/*.json                        ← 100 games/role
├── 300games/
│   ├── vs_base/*.json                    ← 300 games/role
│   └── vs_self/*.json                    ← 300 games/role
└── logs/                                 ← raw eval driver logs
```

Each JSON contains full per-game transcripts (`games.*.investor_text`, `games.*.trustee_text`) plus the aggregate `metrics` block.

## Reproducing

```bash
# 100 games/role (default)
bash evaluations/run_lagrpo_trust_game_eval.sh

# 300 games/role, separate output dir
NUM_GAMES=300 RESULTS_TAG=300games bash evaluations/run_lagrpo_trust_game_eval.sh
```
