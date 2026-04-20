#!/bin/bash
# RQ5 — Capability benchmarks (MMLU-Pro, IFEval, GSM8K) on A100 via vLLM.
# Companion to run_capabilities_eval.sh (HF backend, 5090-targeted, slower).
# See documents/2026-04-19-capabilities-eval-a100-handoff.md for full context.
set -e

cd /workspace/multiturn-llm-training

BASE_MODEL="OpenPipe/Qwen3-14B-Instruct"
OUTPUT_DIR="output"
# Merged models go on local overlay SSD (56GB free, fast) instead of mfs workspace
# (workspace has a ~20GB user quota — a single 28GB merged model won't fit).
MERGED_DIR="/root/merged"
RESULTS_DIR="evaluations/results/capabilities"
CHECKPOINTS_MD="$RESULTS_DIR/checkpoints.md"
TASKS="mmlu_pro,ifeval,gsm8k"

# 5-shot MMLU-Pro (published-comparable). No --limit, full benchmark.
COMMON_ARGS="--tasks $TASKS --batch_size auto --apply_chat_template --log_samples --seed 42"
VLLM_ARGS_BASE="dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=8192,tensor_parallel_size=1"

mkdir -p "$RESULTS_DIR" "$MERGED_DIR"

# Which runs to evaluate. Must match entries in checkpoints.md.
SELECTED_RUNS="grpo_equal lagrpo_equal"

mapfile -t ALL_CHECKPOINTS < <(grep -E '^[a-z_]+\|migub/[a-z0-9_-]+\|[0-9]+$' "$CHECKPOINTS_MD")
CHECKPOINTS=()
for entry in "${ALL_CHECKPOINTS[@]}"; do
    IFS='|' read -r run_id _ _ <<< "$entry"
    for sel in $SELECTED_RUNS; do
        if [ "$run_id" = "$sel" ]; then
            CHECKPOINTS+=("$entry")
            break
        fi
    done
done
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint entries matched SELECTED_RUNS ($SELECTED_RUNS) in $CHECKPOINTS_MD" >&2
    exit 1
fi
echo "Loaded ${#CHECKPOINTS[@]} checkpoints: ${CHECKPOINTS[*]}"

echo "============================================"
echo "Step 1: Download adapters (if missing)"
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
echo "Step 2: Baseline (base model, vLLM)"
echo "============================================"

if [ -f "$RESULTS_DIR/base.json" ]; then
    echo "[SKIP] base.json already exists"
else
    lm_eval --model vllm \
        --model_args "pretrained=$BASE_MODEL,$VLLM_ARGS_BASE" \
        $COMMON_ARGS \
        --output_path "$RESULTS_DIR/base.json"
fi

echo ""
echo "============================================"
echo "Step 3: Merge + eval each adapter (sequential to save disk)"
echo "============================================"
# Interleaved: merge → eval → delete. Only one merged model on disk at any time.
# Merge uses GPU (device_map=cuda) for ~30s merges vs ~10min on CPU.

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r run_id repo step <<< "$entry"
    ckpt_path="$OUTPUT_DIR/$(basename "$repo")/checkpoint-$step"
    merged_path="$MERGED_DIR/$run_id"
    out_file="$RESULTS_DIR/$run_id.json"

    if [ -f "$out_file" ]; then
        echo "[SKIP] $run_id already evaluated"
        continue
    fi

    if [ ! -f "$merged_path/config.json" ]; then
        echo "[MERGE] $run_id: $ckpt_path -> $merged_path"
        python - <<PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "$BASE_MODEL", torch_dtype=torch.bfloat16, device_map="cuda"
)
model = PeftModel.from_pretrained(base, "$ckpt_path")
merged = model.merge_and_unload()
merged.save_pretrained("$merged_path", safe_serialization=True, max_shard_size="5GB")
tok = AutoTokenizer.from_pretrained("$BASE_MODEL")
tok.save_pretrained("$merged_path")
print("Merged -> $merged_path")
PYEOF
    else
        echo "[SKIP MERGE] $run_id already at $merged_path"
    fi

    echo ""
    echo ">>> Evaluating: $run_id ($merged_path)"
    lm_eval --model vllm \
        --model_args "pretrained=$merged_path,$VLLM_ARGS_BASE" \
        $COMMON_ARGS \
        --output_path "$out_file"
    echo "[DONE] $run_id -> $out_file"

    echo "[CLEANUP] removing $merged_path to free overlay disk"
    rm -rf "$merged_path"
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
