# Multiturn LLM Training — Cooperative Negotiation Extension

This repository extends the original multi-turn LLM training framework by [Luca Franceschetti](https://github.com/LucaFranceschetti) from **competitive (zero-sum)** to **cooperative** negotiation. It is the codebase for the Master's thesis:

**From Competition to Cooperation: Teaching LLMs Pareto-Efficient Negotiation with LA-GRPO**
Michael Gubler, HSLU MSc in Applied Information and Data Science, 2026.

The work builds on:
[Using the Advantage: Teaching LLMs to Negotiate in Multi-Turn Dialogues with Local Advantage GRPO](https://drive.google.com/file/d/1LnP1im742NEbg5YeG_Icjx1kq-dJERdV/view?usp=sharing) — Luca Franceschetti, 2025.

---

## What's New in This Fork

- **Cooperative reward design** combining self-interest, social welfare ($U_A + U_B$), and Nash product ($U_A \times U_B$), each game-normalized.
- **Five $\lambda$ reward configurations** trained with GRPO: self-only, fair-only, welfare-only, all-equal, self-fair-equal.
- **Three configurations** trained additionally with LA-GRPO: self-only, fair-only, all-equal.
- **Base model upgraded** from LLaMA-3.1-8B to Qwen3-14B-Instruct with 4-bit nf4 quantization.
- **Extended LAMEN game suite**: 5 scenarios (rental, loan, merger, joint venture, employment contract), 19 issues, 47 game configurations spanning 7 negotiation archetypes.
- **Single-GPU training pipeline** (rewritten for RTX 5090 / A100 80GB).
- **Trust Game evaluation** for out-of-domain cooperative transfer.
- **Adversarial persona robustness suite** (hardball, deceptive, anchoring, stubborn).
- **Capability benchmarks** (MMLU-Pro, IFEval, GSM8K).

---

## Methods Implemented

- **GRPO** — https://arxiv.org/pdf/2501.12948
- **LA-GRPO** — Franceschetti (2025), [thesis link](https://drive.google.com/file/d/1LnP1im742NEbg5YeG_Icjx1kq-dJERdV/view?usp=sharing)

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (training was done on a single A100 80GB; evaluation on RTX 5090 32GB)
- `secrets.json` file in the repository root with API keys for the judge model:

```json
{
  "openai": {
    "api_key": "your-openai-api-key"
  }
}
```

This file is excluded via `.gitignore` and must be created locally.

### Installation

```bash
git clone https://github.com/migub/multiturn-llm-training.git
cd multiturn-llm-training
bash setup.sh
```

### WandB / HuggingFace authentication

```bash
wandb login
huggingface-cli login
```

---

## Training

### GRPO

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file=$ACCELERATE_CONFIG \
  multiturn_llm_training/GRPO/grpo.py \
  --game-type multi-game \
  --model-name OpenPipe/Qwen3-14B-Instruct \
  --lambda-self 1.0 --lambda-welfare 1.0 --lambda-fair 1.0 \
  --use-wandb
```

Key arguments:

- `--lambda-self`, `--lambda-welfare`, `--lambda-fair` — reward weights
- `--game-type` — `multi-game` (5 scenarios) or `out-of-domain` (Rio Copa)
- `--num-generations` — group size $G$ (default 8)
- `--max-rounds` — turns per agent (default 5)

### LA-GRPO

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file=$ACCELERATE_CONFIG \
  multiturn_llm_training/GRPO/la_grpo.py \
  --game-type multi-game \
  --model-name OpenPipe/Qwen3-14B-Instruct \
  --lambda-self 0.0 --lambda-welfare 0.0 --lambda-fair 1.0 \
  --use-wandb
```

Same arguments as GRPO; the only difference is turn-level credit assignment.

---

## Evaluation

### Negotiation evaluation

```bash
python evaluations/run_negotiation_eval.py \
  --checkpoint output/grpo-multigame-all-equal/checkpoint-620 \
  --num-games 28 --repetitions 20 \
  --output-dir evaluations/results/negotiation/20repetitions
```

### Trust Game evaluation

```bash
python evaluations/run_trust_game_eval.py \
  --checkpoint <path> \
  --num-games 50 \
  --output-dir evaluations/results/trustgame
```

### Robustness evaluation (adversarial personas)

```bash
bash evaluations/run_lagrpo_fair_only_robustness.sh
```

### Capability benchmarks

```bash
lm_eval --model vllm --model_args pretrained=<checkpoint_path> \
  --tasks mmlu_pro,ifeval,gsm8k --batch_size auto
```

---

## Training Logs

Full WandB reports for all training runs are publicly available:

- **GRPO runs**: https://api.wandb.ai/links/michael-gubler-hochschule-luzern/ig05od0g
- **LA-GRPO runs**: https://api.wandb.ai/links/michael-gubler-hochschule-luzern/dlwxgofj
