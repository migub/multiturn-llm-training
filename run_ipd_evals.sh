#!/bin/bash
set -e
cd /workspace/multiturn-llm-training

SELF_DIR="evaluations/results/ipd/vs_self"
BASE_DIR="evaluations/results/ipd/vs_base"
mkdir -p "$SELF_DIR" "$BASE_DIR"

NUM_GAMES=50
TEMP=1.0
SEED=42

CHECKPOINTS=(
    "none|base_model"
    "output/grpo-multigame-self-only/checkpoint-560|self_only"
    "output/grpo-multigame-fair-only/checkpoint-560|fair_only"
    "output/grpo-multigame-all-equal/checkpoint-620|all_equal"
    "output/grpo-multigame-self-fair-equal/checkpoint-820|self_fair_equal"
)

TOTAL=$(( ${#CHECKPOINTS[@]} * 2 ))
STEP=0

echo "============================================"
echo "PRISONER'S DILEMMA EVALUATIONS"
echo "  ${#CHECKPOINTS[@]} models x 2 modes (vs_self + vs_base)"
echo "  $TOTAL total runs, $NUM_GAMES games each, temp=$TEMP"
echo "============================================"

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r ckpt name <<< "$entry"

    # --- vs_self ---
    STEP=$((STEP + 1))
    echo -e "\n>>> [$STEP/$TOTAL] $name — self-play..."
    python evaluations/prisoners_dilemma_eval.py \
        --checkpoint "$ckpt" \
        --self-play \
        --num-games $NUM_GAMES \
        --temperature $TEMP \
        --seed $SEED \
        --output-json "$SELF_DIR/${name}.json"

    # --- vs_base ---
    STEP=$((STEP + 1))
    echo -e "\n>>> [$STEP/$TOTAL] $name — vs base..."
    python evaluations/prisoners_dilemma_eval.py \
        --checkpoint "$ckpt" \
        --num-games $NUM_GAMES \
        --temperature $TEMP \
        --seed $SEED \
        --output-json "$BASE_DIR/${name}.json"
done

echo -e "\n============================================"
echo "ALL PRISONER'S DILEMMA EVALUATIONS COMPLETE"
echo "============================================"
echo "Self-play results:"
ls -lh "$SELF_DIR/"
echo -e "\nVs-base results:"
ls -lh "$BASE_DIR/"
