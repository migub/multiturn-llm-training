#!/bin/bash
# ============================================================
# RunPod Setup Script for GRPO Negotiation Training
# Usage: bash setup.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Setting up GRPO Negotiation Training Environment"
echo "============================================================"

# ---- API Keys ----
OPENAI_API_KEY="DEIN-OPENAI-KEY"
WANDB_API_KEY="DEIN-WANDB-KEY"

# ---- 1. Clone Repo ----
echo "[1/5] Cloning repository..."
cd /root
if [ -d "multiturn-llm-training" ]; then
    echo "Repo already exists, pulling latest..."
    cd multiturn-llm-training
    git pull
else
    git clone https://github.com/migub/multiturn-llm-training.git
    cd multiturn-llm-training
fi

# ---- 2. Install Dependencies ----
echo "[2/5] Installing dependencies..."
pip install -q --upgrade torch transformers accelerate peft bitsandbytes datasets
pip install -q unsloth hydra-core omegaconf pyyaml openai retry attrs wandb numpy pandas tiktoken
pip install -q git+https://github.com/migub/trl.git --force-reinstall --no-deps

# ---- 3. Install tmux ----
echo "[3/5] Installing tmux..."
apt update -qq && apt install -y -qq tmux > /dev/null 2>&1

# ---- 4. Setup API Keys ----
echo "[4/5] Setting up API keys..."
export OPENAI_API_KEY="$OPENAI_API_KEY"
echo "{\"openai\": {\"api_key\": \"$OPENAI_API_KEY\"}}" > /root/multiturn-llm-training/secrets.json

# ---- 5. Wandb Login ----
echo "[5/5] Logging into Wandb..."
wandb login "$WANDB_API_KEY"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To start training, run:"
echo ""
echo "  tmux new -s training"
echo ""
echo "Then inside tmux:"
echo ""
echo "  cd /root/multiturn-llm-training"
echo ""
echo "  # Run 1: Self-interest only (baseline)"
echo "  python multiturn_llm_training/grpo/grpo_single_gpu.py --use-wandb --game-type multi-game --lambda-self 1.0 --lambda-welfare 0.0 --lambda-fair 0.0 --num-generations 8 --train-size 200 --run-name ablation_self_only --output-dir output/ablation_self_only"
echo ""
echo "  # Run 2: Social welfare only"
echo "  python multiturn_llm_training/grpo/grpo_single_gpu.py --use-wandb --game-type multi-game --lambda-self 0.0 --lambda-welfare 1.0 --lambda-fair 0.0 --num-generations 8 --train-size 200 --run-name ablation_welfare_only --output-dir output/ablation_welfare_only"
echo ""
echo "  # Run 3: Nash product only"
echo "  python multiturn_llm_training/grpo/grpo_single_gpu.py --use-wandb --game-type multi-game --lambda-self 0.0 --lambda-welfare 0.0 --lambda-fair 1.0 --num-generations 8 --train-size 200 --run-name ablation_nash_only --output-dir output/ablation_nash_only"
echo ""
echo "  # Run 4: Equal weights (1/3 each)"
echo "  python multiturn_llm_training/grpo/grpo_single_gpu.py --use-wandb --game-type multi-game --lambda-self 0.333 --lambda-welfare 0.333 --lambda-fair 0.333 --num-generations 8 --train-size 200 --run-name ablation_equal_third --output-dir output/ablation_equal_third"
echo ""
echo "Detach from tmux with: Ctrl+B then D"
echo "Reattach with: tmux attach -t training"