#!/bin/bash
# Run Iterated Stag Hunt evaluations for all model variants.
# Produces both vs-base (trained model vs frozen base) and vs-self (self-play)
# results for the base model and the three trained checkpoints.

set -e
cd /workspace/multiturn-llm-training

VS_BASE_DIR="evaluations/results/staghunt/vs_base"
VS_SELF_DIR="evaluations/results/staghunt/vs_self"
mkdir -p "$VS_BASE_DIR" "$VS_SELF_DIR"

NUM_GAMES=50
NUM_ROUNDS=10
TEMP=1.0
SEED=42

# Checkpoints: (tag, path)
declare -a CHECKPOINTS=(
    "base_model|none"
    "self_only|output/grpo-multigame-self-only/checkpoint-560"
    "fair_only|output/grpo-multigame-fair-only/checkpoint-560"
    "all_equal|output/grpo-multigame-all-equal/checkpoint-620"
)

echo "============================================"
echo "ITERATED STAG HUNT EVALUATIONS"
echo "  Games per variant:   $NUM_GAMES"
echo "  Rounds per game:     $NUM_ROUNDS"
echo "  Temperature:         $TEMP"
echo "  Seed:                $SEED"
echo "============================================"

# ---- vs frozen base model (trained LoRA on for one role) ----
echo -e "\n############################################"
echo "### PHASE 1: vs frozen base model"
echo "############################################"

idx=1
total=${#CHECKPOINTS[@]}
for entry in "${CHECKPOINTS[@]}"; do
    tag="${entry%%|*}"
    ckpt="${entry##*|}"
    echo -e "\n>>> [vs_base $idx/$total] $tag  ($ckpt)"
    python evaluations/stag_hunt_eval.py \
        --checkpoint "$ckpt" \
        --num-games $NUM_GAMES \
        --num-rounds $NUM_ROUNDS \
        --temperature $TEMP \
        --seed $SEED \
        --output-json "$VS_BASE_DIR/${tag}.json"
    idx=$((idx + 1))
done

# ---- self-play (trained LoRA on for both roles) ----
echo -e "\n############################################"
echo "### PHASE 2: self-play"
echo "############################################"

idx=1
for entry in "${CHECKPOINTS[@]}"; do
    tag="${entry%%|*}"
    ckpt="${entry##*|}"
    echo -e "\n>>> [vs_self $idx/$total] $tag  ($ckpt)"
    python evaluations/stag_hunt_eval.py \
        --checkpoint "$ckpt" \
        --self-play \
        --num-games $NUM_GAMES \
        --num-rounds $NUM_ROUNDS \
        --temperature $TEMP \
        --seed $SEED \
        --output-json "$VS_SELF_DIR/${tag}.json"
    idx=$((idx + 1))
done

echo -e "\n============================================"
echo "ALL STAG HUNT EVALUATIONS COMPLETE"
echo "  vs_base results: $VS_BASE_DIR/"
echo "  vs_self results: $VS_SELF_DIR/"
echo "============================================"
