# Evaluations

Out-of-domain evaluations for trained negotiation models.

## Trust Game

**Script:** `trust_game_eval.py`

Tests whether cooperative negotiation training generalizes to the Trust Game (a 2-turn game-theoretic setting the model was never trained on).

### Setup

```bash
pip install torch transformers accelerate peft bitsandbytes
```

Requires a CUDA GPU (~9-10 GB VRAM for Qwen3-14B in 4-bit).

### Modes

**vs_base** — Trained model plays one role, frozen base model (no LoRA) plays the other. Runs N games as Investor + N games as Trustee.

```bash
python evaluations/trust_game_eval.py \
    --checkpoint output/<run_name>/checkpoint-<step> \
    --num-games 50 \
    --temperature 1.0 \
    --output-json evaluations/results/trustgame/vs_base/<run_name>.json
```

**vs_self** — Trained model plays both roles (self-play, LoRA on for both). Runs N games total.

```bash
python evaluations/trust_game_eval.py \
    --checkpoint output/<run_name>/checkpoint-<step> \
    --self-play \
    --num-games 50 \
    --temperature 1.0 \
    --output-json evaluations/results/trustgame/vs_self/<run_name>.json
```

**Base model baseline** — No LoRA, just the base Qwen3-14B.

```bash
python evaluations/trust_game_eval.py \
    --checkpoint none \
    --num-games 50 \
    --temperature 1.0 \
    --output-json evaluations/results/trustgame/vs_base/base_model.json
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to LoRA checkpoint dir, or `none` for base model |
| `--model-name` | `OpenPipe/Qwen3-14B-Instruct` | Base model |
| `--num-games` | 50 | Games per role (vs_base) or total (vs_self) |
| `--temperature` | 0.6 | Sampling temperature |
| `--self-play` | off | Trained model plays both roles |
| `--verbose` | off | Print full dialogues and payoffs per game |
| `--output-json` | None | Save results to JSON |
| `--seed` | 42 | Random seed |
| `--no-quantized` | off | Disable 4-bit quantization |

### Checkpoints

Trained LoRA adapters are hosted on HuggingFace: [huggingface.co/migub](https://huggingface.co/migub)

**Download all checkpoints at once:**

```bash
pip install huggingface_hub
bash evaluations/download_checkpoints.sh
```

This downloads each model's checkpoint into `output/<model-name>/<checkpoint>/`, skipping any that already exist. See `download_checkpoints.sh` for details.

**Download a single checkpoint manually:**

```bash
huggingface-cli download migub/grpo-multigame-fair-only \
    --include "checkpoint-560/*" \
    --local-dir output/grpo-multigame-fair-only
```

### Available Checkpoints

| Run name | Method | lambda_self | lambda_welfare | lambda_fair | Checkpoint |
|----------|--------|-------------|----------------|-------------|------------|
| `grpo-multigame-self-only` | GRPO | 1.0 | 0.0 | 0.0 | checkpoint-560 |
| `grpo-multigame-fair-only` | GRPO | 0.0 | 0.0 | 1.0 | checkpoint-560 |
| `grpo-multigame-all-equal` | GRPO | 0.33 | 0.33 | 0.33 | checkpoint-620 |
| `grpo-multigame-self-fair-equal` | GRPO | 0.5 | 0.0 | 0.5 | checkpoint-820 |

### Results Structure

```
evaluations/results/trustgame/
  vs_base/           # trained model vs frozen base model
    base_model.json
    self_only.json
    fair_only.json
  vs_self/           # trained model vs itself (self-play)
    ...
```

Each JSON contains:
- `args` — run configuration
- `metrics` — aggregate metrics (avg_send, avg_return, return_ratio, payoffs, social_welfare, nash_product, etc.)
- `games` — raw dialogues and extracted actions for every game

### Reference Points

| Outcome | Send | Return | Investor | Trustee | Social Welfare | Nash Product |
|---------|------|--------|----------|---------|----------------|--------------|
| Nash equilibrium (selfish) | 0 | 0 | 10 | 0 | 10 | 0 |
| Pareto optimum | 10 | 15 | 15 | 15 | 30 | 225 |

---

## Iterated Prisoner's Dilemma

**Script:** `prisoners_dilemma_eval.py`

Tests whether cooperative negotiation training leads to sustained cooperation in the classic multi-round Prisoner's Dilemma — the most studied game for the emergence of cooperation.

### Why this game?

- **Multi-turn by nature** — unlike one-shot games, players make decisions over 10 rounds with full history, matching the multi-turn training setup
- **Large gap between Nash and Pareto** — mutual defection yields (1,1) per round, mutual cooperation yields (3,3). Over 10 rounds: Nash total = (10,10), Pareto total = (30,30)
- **Tests strategy emergence** — does the model learn Tit-for-Tat, Always-Cooperate, or Always-Defect? The script tracks retaliation rate (defect after opponent defected) and forgiveness rate (cooperate after opponent defected)
- **Directly relevant to RQ4** — a cooperatively trained model should cooperate but also retaliate against defection (robustness)

### Payoff Matrix (per round)

|  | Opponent Cooperates | Opponent Defects |
|--|---------------------|------------------|
| **You Cooperate** | 3, 3 | 0, 5 |
| **You Defect** | 5, 0 | 1, 1 |

### Modes

**vs_base** — Trained model plays one role (A or B), frozen base model plays the other. Runs N games per role.

```bash
python evaluations/prisoners_dilemma_eval.py \
    --checkpoint output/<run_name>/checkpoint-<step> \
    --num-games 50 \
    --output-json evaluations/results/ipd/vs_base/<run_name>.json
```

**vs_self** — Trained model plays both roles (self-play, LoRA on for both). Runs N games total.

```bash
python evaluations/prisoners_dilemma_eval.py \
    --checkpoint output/<run_name>/checkpoint-<step> \
    --self-play \
    --num-games 50 \
    --output-json evaluations/results/ipd/vs_self/<run_name>.json
```

**Base model baseline** — No LoRA, just the base Qwen3-14B.

```bash
python evaluations/prisoners_dilemma_eval.py \
    --checkpoint none \
    --num-games 20 \
    --output-json evaluations/results/ipd/vs_base/base_model.json
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to LoRA checkpoint dir, or `none` for base model |
| `--model-name` | `OpenPipe/Qwen3-14B-Instruct` | Base model |
| `--num-games` | 50 | Games per role (vs_base) or total (vs_self) |
| `--num-rounds` | 10 | Rounds per game |
| `--temperature` | 0.6 | Sampling temperature |
| `--self-play` | off | Trained model plays both roles |
| `--verbose` | off | Print round-by-round details per game |
| `--output-json` | None | Save results to JSON |
| `--seed` | 42 | Random seed |
| `--no-quantized` | off | Disable 4-bit quantization |

### Metrics

| Metric | Description |
|--------|-------------|
| Cooperation rate | Fraction of rounds where player chose Cooperate |
| Mutual cooperation rate | Fraction of rounds where both cooperated |
| Mutual defection rate | Fraction of rounds where both defected |
| Retaliation rate | P(defect this round \| opponent defected last round) |
| Forgiveness rate | P(cooperate this round \| opponent defected last round) |
| Per-round cooperation | Cooperation rate at each round (shows if cooperation decays over time) |
| Social welfare | Sum of both players' total payoffs |
| Nash product | Product of both players' total payoffs |

### Reference Points (10 rounds)

| Strategy | Per round | Total A | Total B | Social Welfare | Nash Product |
|----------|-----------|---------|---------|----------------|--------------|
| Mutual defection (Nash) | (1, 1) | 10 | 10 | 20 | 100 |
| Mutual cooperation (Pareto) | (3, 3) | 30 | 30 | 60 | 900 |
| Tit-for-Tat vs Always-Defect | varies | ~14 | ~14 | ~28 | ~196 |

### What to look for

- **Cooperative model** should show high mutual cooperation rate (>70%), high social welfare (>50), and moderate retaliation rate (Tit-for-Tat-like)
- **Selfish model** should show low cooperation, high mutual defection
- **Per-round evolution** reveals if cooperation is stable or collapses in later rounds (end-game defection)
- **Forgiveness rate** shows whether the model can recover from mutual defection spirals
