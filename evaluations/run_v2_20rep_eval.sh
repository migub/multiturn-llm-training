#!/bin/bash
# Re-run multi-game negotiation eval (20 repetitions) for all GRPO + LA-GRPO models.
# - GRPO models  -> checkpoint-560
# - LA-GRPO models -> checkpoint-760
# Results written to evaluations/results/negotiation/v2/
set -e

cd /workspace/multiturn-llm-training

OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/negotiation/v2"
mkdir -p "$RESULTS_DIR"

REPS=20
GAME_TYPE="multi-game"

echo "============================================"
echo "Step 1: Download checkpoints"
echo "============================================"

download_ckpt() {
    local repo="$1"
    local ckpt="$2"
    local target_dir="${OUTPUT_DIR}/${repo}/${ckpt}"
    if [ -f "$target_dir/adapter_model.safetensors" ]; then
        echo "[SKIP] ${repo}/${ckpt} already exists"
    else
        echo "[DOWNLOAD] migub/${repo} -> ${ckpt}"
        hf download "migub/${repo}" --include "${ckpt}/*" --local-dir "${OUTPUT_DIR}/${repo}"
        echo "[OK] ${repo}/${ckpt}"
    fi
}

# GRPO @ 560
download_ckpt "grpo-multigame-self-only"      "checkpoint-560"
download_ckpt "grpo-multigame-fair-only"      "checkpoint-560"
download_ckpt "grpo-multigame-all-equal"      "checkpoint-560"
download_ckpt "grpo-multigame-self-fair-equal" "checkpoint-560"

# LA-GRPO @ 760
download_ckpt "lagrpo-self-only-v2"           "checkpoint-760"
download_ckpt "lagrpo-multigame-all-equal"    "checkpoint-760"
download_ckpt "lagrpo-fair-only"              "checkpoint-760"

echo ""
echo "============================================"
echo "Step 2: Run 20-rep multi-game evaluations"
echo "============================================"

run_eval() {
    local ckpt_path="$1"
    local out_name="$2"
    local lam_self="$3"
    local lam_welfare="$4"
    local lam_fair="$5"

    echo ""
    echo ">>> Evaluating: ${out_name} (${ckpt_path})"
    python evaluations/run_negotiation_eval.py \
        --checkpoint "${ckpt_path}" \
        --repetitions ${REPS} \
        --game-type ${GAME_TYPE} \
        --output-dir "$RESULTS_DIR" \
        --lambda-self ${lam_self} \
        --lambda-welfare ${lam_welfare} \
        --lambda-fair ${lam_fair}

    # run_negotiation_eval.py writes <basename(checkpoint)>.json — rename to descriptive name
    local base
    if [ "$ckpt_path" = "none" ]; then
        base="none"
    else
        base=$(basename "${ckpt_path%/}")
    fi
    if [ -f "$RESULTS_DIR/${base}.json" ] && [ "$base" != "$out_name" ]; then
        mv "$RESULTS_DIR/${base}.json" "$RESULTS_DIR/${out_name}.json"
    fi
    echo "[DONE] ${out_name} -> $RESULTS_DIR/${out_name}.json"
}

# Base model (no LoRA)
run_eval "none" "base_model" 1.0 0.0 0.0

# GRPO @ 560
run_eval "output/grpo-multigame-self-only/checkpoint-560"      "grpo_self_only_560"      1.0  0.0  0.0
run_eval "output/grpo-multigame-fair-only/checkpoint-560"      "grpo_fair_only_560"      0.0  0.0  1.0
run_eval "output/grpo-multigame-all-equal/checkpoint-560"      "grpo_all_equal_560"      0.33 0.33 0.33
run_eval "output/grpo-multigame-self-fair-equal/checkpoint-560" "grpo_self_fair_equal_560" 0.5  0.0  0.5

# LA-GRPO @ 760
run_eval "output/lagrpo-self-only-v2/checkpoint-760"           "lagrpo_self_only_760"    1.0  0.0  0.0
run_eval "output/lagrpo-multigame-all-equal/checkpoint-760"    "lagrpo_all_equal_760"    0.33 0.33 0.33
run_eval "output/lagrpo-fair-only/checkpoint-760"              "lagrpo_fair_only_760"    0.0  0.0  1.0

echo ""
echo "============================================"
echo "All v2 20-rep evaluations complete!"
echo "Results in: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null
