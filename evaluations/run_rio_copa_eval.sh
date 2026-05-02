#!/bin/bash
# Out-of-domain (Rio Copa + Research Collaboration) negotiation eval, 20 reps.
# Two sets:
#   v1 = checkpoints used in evaluations/results/negotiation/20repetitions/
#   v2 = checkpoints used in evaluations/results/negotiation/v2/        (GRPO@560, LA-GRPO@760)
# Shared runs (base, grpo_self_only_560, grpo_fair_only_560) executed once and copied to both folders.
set -e

cd /workspace/multiturn-llm-training

OUTPUT_DIR="output"
RES_V1="evaluations/results/negotiation/rio_copa/v1"
RES_V2="evaluations/results/negotiation/rio_copa/v2"
mkdir -p "$RES_V1" "$RES_V2"

REPS=20
GAME_TYPE="out-of-domain"

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

# v1-only checkpoints
download_ckpt "grpo-multigame-all-equal"       "checkpoint-620"
download_ckpt "grpo-multigame-self-fair-equal" "checkpoint-820"
download_ckpt "lagrpo-self-only-v2"            "checkpoint-2000"
download_ckpt "lagrpo-multigame-all-equal"     "checkpoint-2000"
download_ckpt "lagrpo-fair-only"               "checkpoint-1340"

# v2 + shared (already present from previous run, but check anyway)
download_ckpt "grpo-multigame-self-only"       "checkpoint-560"
download_ckpt "grpo-multigame-fair-only"       "checkpoint-560"
download_ckpt "grpo-multigame-all-equal"       "checkpoint-560"
download_ckpt "grpo-multigame-self-fair-equal" "checkpoint-560"
download_ckpt "lagrpo-self-only-v2"            "checkpoint-760"
download_ckpt "lagrpo-multigame-all-equal"     "checkpoint-760"
download_ckpt "lagrpo-fair-only"               "checkpoint-760"

echo ""
echo "============================================"
echo "Step 2: Run 20-rep out-of-domain evaluations"
echo "============================================"

# Runs once, writes to $1 (output dir), then optionally copies to $2.
run_eval() {
    local out_dir="$1"
    local copy_dir="$2"      # may be empty
    local ckpt_path="$3"
    local out_name="$4"
    local lam_self="$5"
    local lam_welfare="$6"
    local lam_fair="$7"

    local target="$out_dir/${out_name}.json"
    if [ -f "$target" ]; then
        echo ""
        echo "[SKIP] ${out_name} (already at $target)"
    else
        echo ""
        echo ">>> Evaluating: ${out_name} (${ckpt_path}) -> ${out_dir}"
        python evaluations/run_negotiation_eval.py \
            --checkpoint "${ckpt_path}" \
            --repetitions ${REPS} \
            --game-type ${GAME_TYPE} \
            --output-dir "$out_dir" \
            --lambda-self ${lam_self} \
            --lambda-welfare ${lam_welfare} \
            --lambda-fair ${lam_fair}

        local base
        if [ "$ckpt_path" = "none" ]; then
            base="none"
        else
            base=$(basename "${ckpt_path%/}")
        fi
        if [ -f "$out_dir/${base}.json" ] && [ "$base" != "$out_name" ]; then
            mv "$out_dir/${base}.json" "$target"
        fi
        echo "[DONE] ${out_name} -> $target"
    fi

    if [ -n "$copy_dir" ]; then
        mkdir -p "$copy_dir"
        cp "$target" "$copy_dir/${out_name}.json"
        echo "[COPY] $target -> $copy_dir/${out_name}.json"
    fi
}

# ---- Shared runs (write to v1, copy to v2) ----
run_eval "$RES_V1" "$RES_V2" "none"                                          "base_model"          1.0  0.0  0.0
run_eval "$RES_V1" "$RES_V2" "output/grpo-multigame-self-only/checkpoint-560" "grpo_self_only_560" 1.0  0.0  0.0
run_eval "$RES_V1" "$RES_V2" "output/grpo-multigame-fair-only/checkpoint-560" "grpo_fair_only_560" 0.0  0.0  1.0

# ---- v1-only runs ----
run_eval "$RES_V1" "" "output/grpo-multigame-all-equal/checkpoint-620"       "grpo_all_equal_620"       0.33 0.33 0.33
run_eval "$RES_V1" "" "output/grpo-multigame-self-fair-equal/checkpoint-820" "grpo_self_fair_equal_820" 0.5  0.0  0.5
run_eval "$RES_V1" "" "output/lagrpo-self-only-v2/checkpoint-2000"           "lagrpo_self_only_2000"    1.0  0.0  0.0
run_eval "$RES_V1" "" "output/lagrpo-multigame-all-equal/checkpoint-2000"    "lagrpo_all_equal_2000"    0.33 0.33 0.33
run_eval "$RES_V1" "" "output/lagrpo-fair-only/checkpoint-1340"              "lagrpo_fair_only_1340"    0.0  0.0  1.0

# ---- v2-only runs ----
run_eval "$RES_V2" "" "output/grpo-multigame-all-equal/checkpoint-560"       "grpo_all_equal_560"       0.33 0.33 0.33
run_eval "$RES_V2" "" "output/grpo-multigame-self-fair-equal/checkpoint-560" "grpo_self_fair_equal_560" 0.5  0.0  0.5
run_eval "$RES_V2" "" "output/lagrpo-self-only-v2/checkpoint-760"            "lagrpo_self_only_760"     1.0  0.0  0.0
run_eval "$RES_V2" "" "output/lagrpo-multigame-all-equal/checkpoint-760"     "lagrpo_all_equal_760"     0.33 0.33 0.33
run_eval "$RES_V2" "" "output/lagrpo-fair-only/checkpoint-760"               "lagrpo_fair_only_760"     0.0  0.0  1.0

echo ""
echo "============================================"
echo "All Rio Copa (out-of-domain) 20-rep evaluations complete!"
echo "v1 results: $RES_V1/"
echo "v2 results: $RES_V2/"
echo "============================================"
ls -la "$RES_V1"/*.json "$RES_V2"/*.json 2>/dev/null
