# Out-of-Domain Evaluation Games

## Purpose

These games are used exclusively for **out-of-domain evaluation** — testing whether a model trained on the in-domain scenarios (rental agreement, loan, merger, joint venture, employment contract) can generalize to unseen negotiation domains.

They are **not** used during training. The model never sees these scenarios until evaluation, making them a true test of transfer learning and cooperative negotiation generalization.

## Games

### Rio Copa (Luca's original)
- **File:** `rio_copa.yaml`
- **Scenario:** CPC International acquiring Rio Copa Foods, a family-owned company in Santa Cruz
- **Issues:** contingent liability (compatible), family employees (integrative), financing terms (integrative), non-compete period (distributive)
- **Source:** Classic negotiation case study used in academic courses
- **Note:** Uses larger payoff scales (0–2500) compared to other games

### Research Collaboration (Michael, new)
- **File:** `research-collaboration.yaml`
- **Scenario:** A university research lab and a pharma company negotiating a joint drug development partnership
- **Issues:**
  - `rc-funding.yaml` — Annual research funding, $1M–$11M (compatible)
  - `rc-ip-ownership.yaml` — IP ownership share, 0%–100% (distributive)
  - `rc-publication-rights.yaml` — Publication embargo, none–18 months (integrative)
  - `rc-data-access.yaml` — Data sharing level, minimal–full integration (compatible)
  - `rc-project-duration.yaml` — Project length, 1–6 years (integrative)
- **Why this domain:** Academia vs. industry creates natural tension between openness (publications, data) and commercial protection (IP, speed to market). Tests all three issue archetypes in a context very different from business-to-business deals.

## How to Run

Evaluate a trained checkpoint on out-of-domain scenarios:

```bash
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --game-type out-of-domain \
    --model-name OpenPipe/Qwen3-14B-Instruct \
    --resume-from-checkpoint output/<run_name>/checkpoint-<N> \
    --use-wandb \
    --eval-steps 1 \
    --save-steps 999 \
    --train-size 100 \
    --run-name eval_ood_<run_name>
```

Replace `<run_name>` and `<N>` with your trained model's run name and checkpoint number. Check available checkpoints with `ls output/`.

## Eval Dataset

The `out-of-domain` eval dataset contains 10 curated configs (5 per game):

| # | Game | Issues | Archetype |
|---|------|--------|-----------|
| 1 | Rio Copa | contingent liability | Single compatible |
| 2 | Rio Copa | family employees | Single integrative |
| 3 | Rio Copa | financing + non-compete | Integrative + distributive |
| 4 | Rio Copa | contingent liability + family employees | Compatible + integrative |
| 5 | Rio Copa | financing + family employees | Integrative + integrative |
| 6 | Research Collab | funding | Single compatible |
| 7 | Research Collab | IP ownership | Single distributive |
| 8 | Research Collab | publication rights + project duration | Integrative + integrative |
| 9 | Research Collab | data access + IP ownership | Compatible + distributive |
| 10 | Research Collab | publication rights + funding | Integrative + compatible |
