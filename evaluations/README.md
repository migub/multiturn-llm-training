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

Download a checkpoint:

```bash
# Install huggingface CLI if needed
pip install huggingface_hub

# GRPO checkpoints
huggingface-cli download migub/grpo-multigame-self-only --local-dir output/grpo-multigame-self-only/checkpoint
huggingface-cli download migub/grpo-multigame-fair-only --local-dir output/grpo-multigame-fair-only/checkpoint
huggingface-cli download migub/grpo-multigame-all-equal --local-dir output/grpo-multigame-all-equal/checkpoint

# LA-GRPO checkpoints
huggingface-cli download migub/lagrpo-multigame-self-only --local-dir output/lagrpo-multigame-self-only/checkpoint
huggingface-cli download migub/lagrpo-multigame-fair-only --local-dir output/lagrpo-multigame-fair-only/checkpoint
huggingface-cli download migub/lagrpo-multigame-all-equal --local-dir output/lagrpo-multigame-all-equal/checkpoint
```

Each HuggingFace repo contains the most recent checkpoint (LoRA adapter files). `huggingface-cli download` always pulls the latest version.

All models at: [huggingface.co/migub](https://huggingface.co/migub)

### Run Names

Available checkpoints:

| Run name | Method | lambda_self | lambda_fair |
|----------|--------|-------------|-------------|
| `grpo-multigame-self-only` | GRPO | 1.0 | 0.0 |
| `grpo-multigame-fair-only` | GRPO | 0.0 | 1.0 |
| `grpo-multigame-all-equal` | GRPO | 0.33 | 0.33 |
| `lagrpo-multigame-self-only` | LA-GRPO | 1.0 | 0.0 |
| `lagrpo-multigame-fair-only` | LA-GRPO | 0.0 | 1.0 |
| `lagrpo-multigame-all-equal` | LA-GRPO | 0.33 | 0.33 |

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
