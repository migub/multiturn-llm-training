#!/bin/bash
# Download the latest LoRA checkpoints from HuggingFace for evaluation.
#
# Usage:
#   bash evaluations/download_checkpoints.sh
#
# Requires: pip install huggingface_hub

set -e

OUTPUT_DIR="output"

declare -A MODELS
MODELS=(
    ["grpo-multigame-self-only"]="checkpoint-560"
    ["grpo-multigame-fair-only"]="checkpoint-560"
    ["grpo-multigame-all-equal"]="checkpoint-620"
    ["grpo-multigame-self-fair-equal"]="checkpoint-820"
)

for model in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$model]}"
    target_dir="${OUTPUT_DIR}/${model}/${checkpoint}"

    if [ -d "$target_dir" ] && [ -f "$target_dir/adapter_model.safetensors" ]; then
        echo "[SKIP] ${model}/${checkpoint} already exists"
        continue
    fi

    echo "[DOWNLOAD] migub/${model} -> ${target_dir}"
    huggingface-cli download "migub/${model}" \
        --include "${checkpoint}/*" \
        --local-dir "${OUTPUT_DIR}/${model}"
    echo "[OK] ${model}/${checkpoint}"
    echo
done

echo "All checkpoints downloaded to ${OUTPUT_DIR}/"
echo
echo "Usage with eval scripts:"
echo "  python evaluations/trust_game_eval.py --checkpoint output/grpo-multigame-fair-only/checkpoint-560 --self-play --num-games 50"
echo "  python evaluations/prisoners_dilemma_eval.py --checkpoint output/grpo-multigame-fair-only/checkpoint-560 --self-play --num-games 50"
