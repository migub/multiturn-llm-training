# Current State of the Master Thesis

**Last updated:** 2026-03-31
**Phase:** End of Phase 3 (Training & Experimentation) — moving into Phase 4 (Evaluation & Analysis)
**Deadline:** 2026-05-29

---

## What Has Been Done

### Phase 1: Preparation (Oct–Nov 2025)

- Literature review completed
- Reproduced Franceschetti's LA-GRPO baseline
- Set up computational environment (Google Colab, GitHub, wandb)

### Phase 2: Environment Design (Dec 2025–Jan 2026)

- Extended LAMEN with 2 new scenarios: **Joint Venture** (4 issues) and **Employment Contract** (5 issues)
- Total: 5 scenarios, 19 issues, 48 game combinations (Luca had: 3 scenarios, 10 issues, 23 games)
- Implemented cooperative reward function: `R_coop = lambda_self * ratio_self + lambda_welfare * ratio_welfare + lambda_fair * ratio_nash`
- All components normalized to [0, 1] before weighting (see `documents/2026-03-22-lambda-normalization-and-combinations.md`)
- Added `compute_max_metrics()` to brute-force per-game maxima for normalization
- Added `create_eval_dataset()` for deterministic, comparable evaluation across runs
- Added `cooperative-only` game type (JV + EC only) for ablation
- Added `out-of-domain` game type (rio copa) for generalization eval

### Phase 3: Training Runs (Feb–Mar 2026)

#### Model

- **Qwen3-14B-Instruct** (switched from LLaMA 3.1 8B as planned in preliminary study)
- 4-bit quantization (BitsAndBytes nf4, double quant, bf16 compute)
- LoRA: r=8, alpha=16, dropout=0.1, all-linear target modules
- GPU: A100 80GB (Google Colab) / RTX 5090 32GB

#### Completed Runs (GRPO, multi-game) — all 3 crashed but ran 600+ steps

| Run | Wandb ID | lambda_self | lambda_welfare | lambda_fair | Steps | Runtime |
|-----|----------|-------------|----------------|-------------|-------|---------|
| Self-only | `eunj3abz` | 1.0 | 0.0 | 0.0 | 693 | 37.3h |
| Fair-only (Nash) | `gvrgl5tx` | 0.0 | 0.0 | 1.0 | 676 | 31.9h |
| All-equal | `fyt3rskk` | 1/3 | 1/3 | 1/3 | 745 | 66.9h |

Wandb project: `michael-gubler-hochschule-luzern/huggingface`

#### LA-GRPO Implementation

- Turn-level credit assignment implemented in `multiturn_grpo_trainer.py`
- Uses geometric distribution for turn sampling (p=0.3)
- `--turn-level-sampling` flag added
- **LA-GRPO runs not yet started** — implementation is ready

---

## Wandb Results: Detailed Metrics

### Final Metrics (average of last 20 logged steps, ~100 steps window)

| Metric | Self-only | Fair-only | All-equal | Best |
|--------|-----------|-----------|-----------|------|
| **ratio_self** | **0.599** (0.097) | 0.595 (0.068) | 0.591 (0.118) | Self (marginal) |
| **ratio_welfare** | 0.691 (0.081) | **0.761** (0.061) | 0.737 (0.085) | Fair |
| **ratio_nash** | 0.537 (0.103) | **0.610** (0.085) | 0.577 (0.105) | Fair |
| **agreement_rate** | 0.865 (0.063) | **0.930** (0.047) | 0.904 (0.072) | Fair |
| **U_A** | 59.8 (9.7) | 59.4 (6.8) | 59.0 (11.8) | Similar |
| **U_B** | 52.8 (10.9) | **63.9** (7.2) | 56.8 (10.7) | Fair |
| **social_welfare** | 112.7 (18.0) | **123.3** (9.8) | 115.9 (20.6) | Fair |
| **nash_product** | 3716 (1114) | **4141** (729) | 3690 (1284) | Fair |
| **KL divergence** | 0.074 (0.020) | 1.202 (1.579) | **0.050** (0.008) | All-equal (lowest) |

*Values in parentheses are standard deviations across the 20-step window.*

### Training Progression (early → late)

| Metric | Self-only (step 1→682) | Fair-only (step 1→676) | All-equal (step 1→745) |
|--------|------------------------|------------------------|------------------------|
| ratio_self | 0.478 → 0.576 | 0.433 → 0.759 | 0.413 → 0.349 |
| ratio_nash | 0.565 → 0.697 | 0.513 → 0.572 | 0.446 → 0.318 |
| ratio_welfare | 0.729 → 0.794 | 0.700 → 0.744 | 0.581 → 0.592 |
| agreement_rate | 0.900 → 0.875 | 0.875 → 0.950 | 0.725 → 0.825 |
| KL | 0.0003 → 0.088 | 0.0002 → 0.327 | 0.0005 → 0.045 |

*Note: last-step values are noisy. The averaged metrics above are more reliable.*

### Per-Archetype Breakdown (final summary values)

#### Self-only (`eunj3abz`)

| Archetype | Count | U_A | U_B | ratio_self | ratio_nash | ratio_welfare |
|-----------|-------|-----|-----|------------|------------|---------------|
| single-compatible | 16 | 81.9 | 81.9 | 0.819 | 0.732 | 0.819 |
| single-distributive | 8 | 52.5 | 10.0 | 0.525 | 0.250 | 0.625 |
| single-integrative | 8 | 42.5 | 73.8 | 0.425 | 0.930 | 0.969 |
| non-integrative compatible | 24 | 65.0 | 54.6 | 0.650 | 0.663 | 0.782 |
| non-integrative distributive | 8 | 83.1 | 27.0 | 0.831 | 0.632 | 0.933 |
| integrative compatible | 8 | 59.8 | 80.0 | 0.598 | 0.677 | 0.822 |
| integrative distributive | 24 | 48.2 | 33.0 | 0.482 | 0.393 | 0.580 |

#### Fair-only (`gvrgl5tx`)

| Archetype | Count | U_A | U_B | ratio_self | ratio_nash | ratio_welfare |
|-----------|-------|-----|-----|------------|------------|---------------|
| single-compatible | 16 | 84.4 | 84.4 | 0.844 | 0.766 | 0.844 |
| single-distributive | 8 | 76.3 | 23.8 | 0.763 | 0.695 | 1.000 |
| single-integrative | 8 | 27.5 | 86.3 | 0.275 | 0.727 | 0.948 |
| non-integrative compatible | 24 | 65.8 | 67.1 | 0.658 | 0.763 | 0.868 |
| non-integrative distributive | 8 | 70.9 | 41.5 | 0.709 | 0.820 | 0.952 |
| integrative compatible | 40 | 75.9 | 67.3 | 0.759 | 0.572 | 0.744 |
| integrative distributive | 24 | 59.4 | 43.3 | 0.594 | 0.486 | 0.734 |

#### All-equal (`fyt3rskk`)

| Archetype | Count | U_A | U_B | ratio_self | ratio_nash | ratio_welfare |
|-----------|-------|-----|-----|------------|------------|---------------|
| single-compatible | 8 | 78.8 | 78.8 | 0.788 | 0.644 | 0.788 |
| single-distributive | 8 | 22.5 | 52.5 | 0.225 | 0.550 | 0.750 |
| single-integrative | 8 | 47.5 | 45.0 | 0.475 | 0.683 | 0.841 |
| non-integrative compatible | 24 | 67.7 | 68.3 | 0.677 | 0.778 | 0.887 |
| non-integrative distributive | 8 | 49.4 | 45.9 | 0.494 | 0.584 | 0.807 |
| integrative compatible | 8 | 82.5 | 22.8 | 0.825 | 0.371 | 0.702 |
| integrative distributive | 24 | 23.1 | 59.7 | 0.231 | 0.224 | 0.503 |

### Agreed-Only Metrics (quality of successful negotiations)

| Metric | Self-only | Fair-only | All-equal |
|--------|-----------|-----------|-----------|
| agreed/ratio_self | 0.702 | 0.799 | 0.423 |
| agreed/ratio_welfare | 0.867 | 0.783 | 0.718 |
| agreed/ratio_nash | 0.666 | 0.602 | 0.386 |
| agreed/social_welfare | 128.2 | 150.7 | 103.9 |

---

## Key Findings So Far

### 1. Fair-only (Nash product) consistently achieves highest agreement rate

Agreement rate: **0.930** (fair-only) vs 0.904 (all-equal) vs 0.865 (self-only). Nash product reward penalizes failures (0 x anything = 0), so the agent learns to always reach a deal. **Strongest differentiator and key finding for RQ2.**

### 2. Fair-only produces the most balanced and efficient deals

Fair-only dominates on ratio_welfare (0.761), ratio_nash (0.610), social_welfare (123.3), and nash_product (4141). It also gives the opponent significantly more (U_B=63.9 vs 52.8/56.8), creating more total value.

### 3. All three runs achieve similar self-interest (U_A ~59)

Despite different reward functions, U_A converges to ~59 across all runs. The key difference is how much value the opponent gets and how often deals are reached.

### 4. All-equal run has lowest KL divergence (most stable)

KL: 0.050 (all-equal) vs 0.074 (self-only) vs **1.202** (fair-only). Fair-only's KL is dangerously high (std=1.58), suggesting instability. The combined reward in all-equal provides more stable gradient signal.

### 5. Fair-only has high KL variance — potential instability

Fair-only KL reached 1.2+ average with std of 1.58 in the last 100 steps. This is a warning sign — the model is diverging significantly from the reference policy. May need higher beta (KL penalty) for pure Nash runs.

### 6. All-equal underperforms expectations — explained by literature

Despite combining all three objectives, all-equal doesn't clearly outperform either extreme. Its ratio_nash (0.577) is between self-only (0.537) and fair-only (0.610), and its agreed-only metrics are the worst. Literature analysis (see `documents/2026-03-31-lambda-selection-literature-review.md`) explains why: (a) welfare provides zero signal in distributive games, diluting 1/3 of the reward; (b) Nash product already captures efficiency + fairness (Caragiannis et al., 2019), making welfare partially redundant; (c) self-interest at 1/3 is too weak for stabilization (Peysakhovich & Lerer, 2017 suggest ≥30–50%).

### 7. Per-archetype insights

- **Single-distributive:** Self-only extracts much more (U_A=52.5) vs all-equal (22.5) and fair-only (76.3). Fair-only achieves ratio_welfare=1.0 (maximum possible welfare in distributive games).
- **Integrative-distributive:** All runs struggle here (lowest ratios). This archetype requires the most sophisticated trading strategy.
- **Non-integrative compatible:** All runs do reasonably well. Compatible issues are "easy" — both sides prefer the same outcome.
- **Single-integrative:** Self-only gets low U_A (42.5) but high ratio_nash (0.930) — integrative issues naturally balance outcomes.

### 8. Reward normalization is critical

Without normalization, the welfare term dominates due to raw scale differences (0–200 vs 0–100). After normalization, all components contribute proportionally to their lambda weights. See `documents/2026-03-22-lambda-normalization-and-combinations.md`.

### 9. Temperature 0.6 caused mode collapse

Qwen3's default temperature=0.6 led to identical GRPO generations (no diversity = no learning signal). Fixed by setting temperature=1.0. See `documents/2026-03-21-temperature-mode-collapse.md`.

### 10. Critical bugs found and fixed (2026-03-20)

- **`lookup_payoff` substring bug:** `"0%"` matched `"70%"`, corrupting all %-based game rewards
- **TRL loss type default changed:** Was silently using DAPO instead of GRPO (no KL penalty)
- **Wandb metrics only logged last batch** instead of averaging over logging_steps
- See `documents/2026-03-20-code-review-fixes.md` for full list

### 11. Known unfixed issues

- **Partial agreements get nonzero reward:** In 2-issue games, agreeing on 1 issue still gives positive reward (should be 0)
- **Agent leaks payoff info in dialogue:** Model reveals internal payoff values despite system prompt forbidding it (improves with training)
- **Slow sequential generation:** ~80 forward passes per step. vLLM colocate mode planned but not implemented
- **All 3 runs crashed** (state=crashed in wandb) — likely Colab session timeouts, but each ran 600+ steps

---

## Current Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | `OpenPipe/Qwen3-14B-Instruct` |
| Quantization | 4-bit (nf4, double quant) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Learning rate | 5e-5 |
| Beta (KL coefficient) | 0.08 |
| Num generations (G) | 8 |
| Max rounds | 5 |
| Max tokens per turn | 200 |
| Temperature | 1.0 |
| Loss type | grpo |
| Logging steps | 5 |
| Eval steps | 20 |
| Opponent | Frozen local model (LoRA disabled) |

---

## What Comes Next

### Immediate (April 2026)

1. **Literature-backed lambda ablation runs** — See `documents/2026-03-31-lambda-selection-literature-review.md` for full analysis. Primary ablation:
   - Nash-dominant (0.3/0.0/0.7) — Caragiannis et al. (2019): Nash welfare = efficiency + fairness
   - Balanced self+nash (0.5/0.0/0.5) — Peysakhovich & Lerer (2017): ~50% other-regard upper bound
   - Secondary: Nash+welfare (0.3/0.2/0.5), Balanced-all (0.4/0.3/0.3) — tests welfare's contribution
2. **LA-GRPO runs** — Compare standard GRPO vs LA-GRPO for same lambda configs (RQ1)
3. **Fix partial agreement reward** — `get_payoffs()` should return 0 when any issue is N/A
4. **Investigate fair-only KL instability** — Consider higher beta or gradient clipping

### Evaluation (April 2026)

5. **Out-of-domain evaluation** — Rio Copa game (unseen during training)
6. **Robustness testing** — Cooperative agent vs adversarial/selfish opponents (RQ4)
7. **Benchmark tests** — MMLU-Pro, GLUE, IFEval for capability preservation (RQ5)
8. **Pareto frontier analysis** — Plot per-archetype Pareto frontiers (RQ3)

### Writing (May 2026)

9. **Thesis writing** — Results chapter, discussion, conclusion
10. **Deadline: 2026-05-29**

---

## Repository Structure (Key Files)

| File | Role | Status |
|------|------|--------|
| `multiturn_llm_training/grpo/grpo_single_gpu.py` | Training entry point | Active |
| `multiturn_llm_training/grpo/multiturn_grpo_trainer.py` | Multi-turn GRPO/LA-GRPO trainer | Active, LA-GRPO ready |
| `envs/negotiation/env.py` | Cooperative reward, metrics, dataset creation | Active |
| `evaluator/evaluator.py` | Outcome extraction with text-label support | Fixed |
| `envs/negotiation/configs/games/` | 8 game configs (5 scenarios) | Complete |
| `envs/negotiation/configs/issues/` | 19 issue configs | Complete |
| `documents/` | Session notes, findings, implementation plans | 6 documents |

---

## Research Questions → Status

| RQ | Question | Status |
|----|----------|--------|
| RQ1 | Does LA-GRPO improve cooperative negotiation? | LA-GRPO implemented, runs pending |
| RQ2 | How to balance self-interest vs. welfare? | 3 runs done (self-only, fair-only, all-equal). Fair-only best on cooperative metrics. More ablations needed. |
| RQ3 | Which metrics capture cooperative success? | Metrics implemented. Agreement rate is strongest differentiator. Per-archetype analysis available. |
| RQ4 | Robustness against adversarial opponents? | Not started |
| RQ5 | Impact on general LLM capabilities? | Not started |
