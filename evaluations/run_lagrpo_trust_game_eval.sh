#!/bin/bash
# Run Trust Game evaluation for base model + 3 LA-GRPO checkpoints.
# Each model is evaluated in two modes:
#   - vs_base:  trained model as investor vs frozen base, and as trustee vs frozen base
#   - vs_self:  trained model plays both roles (self-play, LoRA on for both)
#
# 100 games per role, temperature 1.0, seed 42. Output JSONs land in
# evaluations/results/trustgame/{vs_base,vs_self}/<name>.json — overwrites
# the existing 50-game baseline runs.

set -e

cd /workspace/multiturn-llm-training

NUM_GAMES="${NUM_GAMES:-100}"
TEMPERATURE=1.0
SEED=42

# Optional tag appended to output dir (e.g. RESULTS_TAG=300games)
RESULTS_TAG="${RESULTS_TAG:-}"
if [ -n "$RESULTS_TAG" ]; then
    BASE_RESULTS="evaluations/results/trustgame/${RESULTS_TAG}"
else
    BASE_RESULTS="evaluations/results/trustgame"
fi
VS_BASE_DIR="${BASE_RESULTS}/vs_base"
VS_SELF_DIR="${BASE_RESULTS}/vs_self"
mkdir -p "$VS_BASE_DIR" "$VS_SELF_DIR"

# Name -> checkpoint path (use "none" for base model)
declare -A CHECKPOINTS=(
    ["base_model"]="none"
    ["lagrpo_self_only"]="output/lagrpo-self-only-v2/checkpoint-2000"
    ["lagrpo_all_equal"]="output/lagrpo-multigame-all-equal/checkpoint-2000"
    ["lagrpo_fair_only"]="output/lagrpo-fair-only/checkpoint-1340"
)

# Stable iteration order (base first, then LA-GRPO in λ order)
ORDER=("base_model" "lagrpo_self_only" "lagrpo_all_equal" "lagrpo_fair_only")

run_eval() {
    local name="$1"
    local ckpt="$2"
    local mode="$3"  # "vs_base" or "vs_self"

    local out_dir
    local extra_flag=""
    if [ "$mode" = "vs_self" ]; then
        out_dir="$VS_SELF_DIR"
        extra_flag="--self-play"
    else
        out_dir="$VS_BASE_DIR"
    fi

    local out_file="${out_dir}/${name}.json"

    echo ""
    echo "============================================"
    echo ">>> [$mode] $name  (checkpoint: $ckpt)"
    echo "============================================"

    python evaluations/trust_game_eval.py \
        --checkpoint "$ckpt" \
        --num-games "$NUM_GAMES" \
        --temperature "$TEMPERATURE" \
        --seed "$SEED" \
        --output-json "$out_file" \
        $extra_flag

    echo "[DONE] $mode/$name -> $out_file"
}

echo "============================================"
echo "Trust Game Evaluation — LA-GRPO + base rerun"
echo "  Games per role:  $NUM_GAMES"
echo "  Temperature:     $TEMPERATURE"
echo "  Seed:            $SEED"
echo "  Models:          ${ORDER[*]}"
echo "============================================"

for name in "${ORDER[@]}"; do
    ckpt="${CHECKPOINTS[$name]}"
    run_eval "$name" "$ckpt" "vs_base"
    run_eval "$name" "$ckpt" "vs_self"
done

echo ""
echo "============================================"
echo "All trust game evaluations complete!"
echo "Results:"
echo "  $VS_BASE_DIR/"
ls -la "$VS_BASE_DIR"/*.json
echo "  $VS_SELF_DIR/"
ls -la "$VS_SELF_DIR"/*.json
echo "============================================"
