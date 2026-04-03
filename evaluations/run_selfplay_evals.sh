#!/bin/bash
# Run selfplay Trust Game evaluations for all model variants
# Each model plays both Investor and Trustee roles with LoRA enabled

set -e
cd /workspace/multiturn-llm-training

RESULTS_DIR="evaluations/results/trustgame/vs_self"
mkdir -p "$RESULTS_DIR"

NUM_GAMES=50
TEMP=1.0
SEED=42

echo "============================================"
echo "SELFPLAY TRUST GAME EVALUATIONS"
echo "============================================"

# 1. Base model (no LoRA)
echo -e "\n>>> [1/4] Base model (no adapter)..."
python evaluations/trust_game_eval.py \
    --checkpoint none \
    --self-play \
    --num-games $NUM_GAMES \
    --temperature $TEMP \
    --seed $SEED \
    --output-json "$RESULTS_DIR/base_model.json"

# 2. Self-only (lambda_self=1.0, lambda_welfare=0, lambda_fair=0)
echo -e "\n>>> [2/4] Self-only model..."
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo-multigame-self-only/checkpoint-560 \
    --self-play \
    --num-games $NUM_GAMES \
    --temperature $TEMP \
    --seed $SEED \
    --output-json "$RESULTS_DIR/self_only.json"

# 3. Fair-only (lambda_self=0, lambda_welfare=0, lambda_fair=1.0)
echo -e "\n>>> [3/4] Fair-only model..."
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo-multigame-fair-only/checkpoint-560 \
    --self-play \
    --num-games $NUM_GAMES \
    --temperature $TEMP \
    --seed $SEED \
    --output-json "$RESULTS_DIR/fair_only.json"

# 4. All-equal (lambda_self=1.0, lambda_welfare=1.0, lambda_fair=1.0)
echo -e "\n>>> [4/4] All-equal model..."
python evaluations/trust_game_eval.py \
    --checkpoint output/grpo-multigame-all-equal/checkpoint-620 \
    --self-play \
    --num-games $NUM_GAMES \
    --temperature $TEMP \
    --seed $SEED \
    --output-json "$RESULTS_DIR/all_equal.json"

echo -e "\n============================================"
echo "ALL SELFPLAY EVALUATIONS COMPLETE"
echo "Results saved to: $RESULTS_DIR/"
echo "============================================"
