# RQ4 Robustness Evaluation — Prompt-Injected Adversaries

**Research question:** *Can cooperative agents remain robust against adversarial opponents?*

The existing 20-rep eval (`evaluations/results/negotiation/20repetitions/`) measured how cooperative checkpoints play against a **frozen base Qwen3-14B** opponent — effectively a cooperative-ish selfplay baseline. RQ4 asks what happens when the opponent is hostile instead.

This eval covers **Tier 2 (T2)** of the robustness plan: prompt-injected LLM adversaries. The same base Qwen3-14B is used for the opponent, but a persona prompt is appended to its system message to push it toward adversarial behavior. No training needed, no extra models loaded — reuses the full selfplay eval pipeline.

Tiers T1 (scripted rule-based opponents) and T3 (trained adversary or GPT-4o under an adversarial prompt) were dropped to keep scope simple. T2 gives a realistic hostile LLM at zero extra compute cost.

---

## What's in place

| File | Role |
|------|------|
| `adversarial_personas.py` | Persona prompt strings and `apply_persona()` helper |
| `run_negotiation_eval.py` | Existing eval script, now with `--opponent-persona` flag |
| `run_robustness_eval.sh` | Loops over all adversarial personas and all default checkpoints |

### Personas

All persona text is appended to the opponent's system prompt (after the game rules) so the behavioral directive is the most salient instruction at generation time.

| Persona | Behavior |
|---------|----------|
| `cooperative` | Baseline — no injection, same as the 20-rep selfplay eval |
| `hardball` | Self-interested, aggressive opening, minimal concessions, willing to walk away |
| `deceptive` | Misrepresents priorities, bluffs about outside offers, fakes reluctance |
| `anchoring` | Opens at the extreme end of the range, concedes in tiny increments |
| `stubborn` | Take-it-or-leave-it ultimatums, refuses to engage with counter-proposals |

All four adversarial personas still obey the hard constraints from the original game prompt (only discuss issues in the payoff table, don't leak internal payoffs).

### What the agent sees

Nothing changes on the agent side. The agent's system prompt, tokens, and LoRA weights are identical to the selfplay eval. Only the opponent's system prompt is modified. This means differences in output metrics are attributable to how each trained checkpoint responds to a more hostile counterparty, not to changes in the agent's own setup.

---

## How to run

### 1. Smoke test — verify personas actually take effect

Prompt injection is brittle with instruction-tuned models. Before committing to the overnight matrix, run one persona with 1 repetition on a single checkpoint and **read a handful of dialogues** to confirm the opponent stays in character.

```bash
python evaluations/run_negotiation_eval.py \
    --opponent-persona hardball \
    --repetitions 1 \
    --num-games 5 \
    --checkpoint output/grpo-multigame-fair-only/checkpoint-560 \
    --output-dir evaluations/results/robustness_smoketest
```

Open the resulting JSON (`evaluations/results/robustness_smoketest/checkpoint-560_hardball.json`) and scan the `conversation` field of a few games. If the opponent still makes cooperative-sounding mid-range offers, the persona isn't taking — either strengthen the prompt text or move the persona block *before* the game rules instead of after.

### 2. Full matrix (overnight)

```bash
bash evaluations/run_robustness_eval.sh
```

Defaults: 4 personas × 5 checkpoints × 14 games × 10 reps ≈ 2800 games. At ~30 s each that's ~24 h on one GPU. Reduce with env vars:

```bash
REPETITIONS=5 bash evaluations/run_robustness_eval.sh                 # half the reps
REPETITIONS=5 bash evaluations/run_robustness_eval.sh hardball stubborn  # subset of personas
```

### 3. Single persona + single checkpoint

```bash
python evaluations/run_negotiation_eval.py \
    --opponent-persona stubborn \
    --repetitions 10 \
    --checkpoint output/grpo-multigame-all-equal/checkpoint-620 \
    --output-dir evaluations/results/robustness
```

---

## Output structure

Results land under `evaluations/results/robustness/` (separate from the `negotiation/` selfplay results so they don't overwrite each other).

```
evaluations/results/robustness/
├── base_model_hardball.json
├── self_only_560_hardball.json
├── fair_only_560_hardball.json
├── all_equal_620_hardball.json
├── self_fair_equal_820_hardball.json
├── base_model_deceptive.json
├── ...
├── evaluation_results_grpo_hardball.csv
├── evaluation_results_grpo_deceptive.csv
├── evaluation_results_grpo_anchoring.csv
└── evaluation_results_grpo_stubborn.csv
```

Each JSON mirrors the selfplay format (`metrics`, `games`, `args`) and the per-persona CSV matches Luca's schema, with one added column `opponent_persona`.

---

## How to read the results

The robustness story lives in the **diff between persona and cooperative baseline**, not absolute numbers. Concretely:

- **Exploitation gap** = `ratio_self(cooperative) − ratio_self(persona)` for the same checkpoint. Low gap = robust agent (doesn't get bullied into giving up payoff). Compare across checkpoints — does `fair-only` collapse under `hardball` more than `self-only` does?
- **Agreement rate under adversary** — cooperative agents *should* walk away from bad deals. An agent that agrees at the same rate against `stubborn` as against a cooperative opponent is being exploited.
- **U_A / U_B asymmetry** — against adversaries, `U_A / U_B` should shift in the adversary's favor for every checkpoint. The question is by how much. If `fair-only` ends up with `U_A ≪ U_B` under `hardball`, cooperative training made it exploitable.
- **Per-archetype breakdown** — integrative games reward agents who search for trades; distributive games are pure splits. Cooperative agents may be robust on one archetype and not the other.

A minimum-viable analysis: for each checkpoint, plot `ratio_self` across {cooperative, hardball, deceptive, anchoring, stubborn} as a grouped bar chart. The slope from left (cooperative) to each adversarial bar is the robustness cost.

---

## Known caveats

- **Prompt injection is brittle.** Qwen3-14B-Instruct was RLHF-ed to be helpful; it may ignore some persona text. Always read raw dialogues before trusting aggregate metrics. If a persona is clearly being ignored, strengthen its prompt or drop it from the analysis.
- **Partial agreement bug still present.** `get_payoffs()` returns a non-zero reward when only one of two issues is agreed, contrary to the game rules. This affects all runs (selfplay and adversarial) equally, so the *diff* is still informative, but absolute agreement rates are optimistic.
- **Same base for agent and opponent.** Using one Qwen3-14B for both roles (with different system prompts) means the adversary has exactly the agent's own reasoning capacity. A GPT-4o adversary would probably be stronger. Consider T3 as a follow-up if T2 results are inconclusive.
- **Temperature = 1.0.** Persona adherence may be more consistent at lower temperatures. Worth a small ablation if personas look noisy in the smoke test.
