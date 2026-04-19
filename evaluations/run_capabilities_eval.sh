#!/bin/bash
# RQ5 — Capability benchmarks (MMLU-Pro, IFEval, GSM8K)
# Runs lm-evaluation-harness on base model + 6 trained adapters.
set -e

cd /workspace/multiturn-llm-training

BASE_MODEL="OpenPipe/Qwen3-14B-Instruct"
OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/capabilities"
CHECKPOINTS_MD="$RESULTS_DIR/checkpoints.md"
TASKS="mmlu_pro,ifeval,gsm8k"
COMMON_ARGS="--tasks $TASKS --batch_size auto --apply_chat_template --log_samples --seed 42"
QUANT_ARGS="load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16,dtype=bfloat16"

mkdir -p "$RESULTS_DIR"

# Source of truth: checkpoints.md. Parse fenced block matching run_id|repo|step.
# Lines look like "grpo_self|migub/grpo-multigame-self-only|560".
mapfile -t CHECKPOINTS < <(grep -E '^[a-z_]+\|migub/[a-z0-9_-]+\|[0-9]+$' "$CHECKPOINTS_MD")
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint entries parsed from $CHECKPOINTS_MD" >&2
    exit 1
fi
echo "Loaded ${#CHECKPOINTS[@]} checkpoints from $CHECKPOINTS_MD"

echo "============================================"
echo "Step 1: Download required checkpoints"
echo "============================================"

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r run_id repo step <<< "$entry"
    local_dir="$OUTPUT_DIR/$(basename "$repo")"
    ckpt_path="$local_dir/checkpoint-$step"
    if [ -f "$ckpt_path/adapter_model.safetensors" ]; then
        echo "[SKIP] $run_id already at $ckpt_path"
    else
        echo "[DOWNLOAD] $repo -> checkpoint-$step"
        hf download "$repo" --include "checkpoint-$step/*" --local-dir "$local_dir"
    fi
done

echo ""
echo "============================================"
echo "Step 2: Baseline (base model)"
echo "============================================"

if [ -f "$RESULTS_DIR/base.json" ]; then
    echo "[SKIP] base.json already exists"
else
    lm_eval --model hf \
        --model_args "pretrained=$BASE_MODEL,$QUANT_ARGS" \
        $COMMON_ARGS \
        --output_path "$RESULTS_DIR/base.json"
fi

echo ""
echo "============================================"
echo "Step 3: Trained adapters (6 runs)"
echo "============================================"

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r run_id repo step <<< "$entry"
    ckpt_path="$OUTPUT_DIR/$(basename "$repo")/checkpoint-$step"
    out_file="$RESULTS_DIR/$run_id.json"

    if [ -f "$out_file" ]; then
        echo "[SKIP] $run_id already evaluated"
        continue
    fi

    echo ""
    echo ">>> Evaluating: $run_id ($ckpt_path)"
    lm_eval --model hf \
        --model_args "pretrained=$BASE_MODEL,peft=$ckpt_path,$QUANT_ARGS" \
        $COMMON_ARGS \
        --output_path "$out_file"
    echo "[DONE] $run_id -> $out_file"
done

echo ""
echo "============================================"
echo "Step 4: Aggregate"
echo "============================================"
python evaluations/scripts/aggregate_capabilities.py \
    --results-dir "$RESULTS_DIR" \
    --output-csv "$RESULTS_DIR/capabilities_summary.csv"

echo ""
echo "============================================"
echo "All capability evaluations complete."
echo "Results in: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.csv 2>/dev/null
