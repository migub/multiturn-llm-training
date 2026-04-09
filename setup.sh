#!/bin/bash
# ============================================================
# RunPod Setup Script for GRPO Negotiation Training
# Run from repo root: bash setup.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Setting up GRPO Negotiation Training Environment"
echo "============================================================"

# ---- 1. Install Dependencies ----
echo "[1/3] Installing dependencies..."
pip install --upgrade torch transformers accelerate peft bitsandbytes datasets
pip install unsloth hydra-core omegaconf pyyaml openai retry attrs wandb numpy pandas tiktoken mergekit llm_blender
pip install unsloth
pip install trl --upgrade --no-deps
pip install huggingface_hub[hf_transfer]

# ---- 2. Install tmux ----
echo "[2/3] Installing tmux..."
apt update && apt install tmux -y

# ---- 3. Done ----
echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  tmux new -s training"
echo "  cd /workspace/multiturn-llm-training"
echo ""
echo "Detach from tmux: Ctrl+B then D"
echo "Reattach: tmux attach -t training"