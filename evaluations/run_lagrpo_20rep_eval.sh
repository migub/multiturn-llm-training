#!/bin/bash
# Download LA-GRPO models and run 20-repetition negotiation evaluation
set -e

cd /workspace/multiturn-llm-training

OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/negotiation/20repetitions"

echo "============================================"
echo "Step 1: Download LA-GRPO checkpoints"
echo "============================================"

# LA-GRPO self-only-v2
if [ -d "$OUTPUT_DIR/lagrpo-self-only-v2/checkpoint-2000" ] && [ -f "$OUTPUT_DIR/lagrpo-self-only-v2/checkpoint-2000/adapter_model.safetensors" ]; then
    echo "[SKIP] lagrpo-self-only-v2/checkpoint-2000 already exists"
else
    echo "[DOWNLOAD] migub/lagrpo-self-only-v2 -> checkpoint-2000"
    hf download migub/lagrpo-self-only-v2 \
        --include "checkpoint-2000/*" \
        --local-dir "$OUTPUT_DIR/lagrpo-self-only-v2"
    echo "[OK] lagrpo-self-only-v2"
fi

# LA-GRPO all-equal
if [ -d "$OUTPUT_DIR/lagrpo-multigame-all-equal/checkpoint-2000" ] && [ -f "$OUTPUT_DIR/lagrpo-multigame-all-equal/checkpoint-2000/adapter_model.safetensors" ]; then
    echo "[SKIP] lagrpo-multigame-all-equal/checkpoint-2000 already exists"
else
    echo "[DOWNLOAD] migub/lagrpo-multigame-all-equal -> checkpoint-2000"
    hf download migub/lagrpo-multigame-all-equal \
        --include "checkpoint-2000/*" \
        --local-dir "$OUTPUT_DIR/lagrpo-multigame-all-equal"
    echo "[OK] lagrpo-multigame-all-equal"
fi

# LA-GRPO fair-only
if [ -d "$OUTPUT_DIR/lagrpo-fair-only/checkpoint-1340" ] && [ -f "$OUTPUT_DIR/lagrpo-fair-only/checkpoint-1340/adapter_model.safetensors" ]; then
    echo "[SKIP] lagrpo-fair-only/checkpoint-1340 already exists"
else
    echo "[DOWNLOAD] migub/lagrpo-fair-only -> checkpoint-1340"
    hf download migub/lagrpo-fair-only \
        --include "checkpoint-1340/*" \
        --local-dir "$OUTPUT_DIR/lagrpo-fair-only"
    echo "[OK] lagrpo-fair-only"
fi

echo ""
echo "============================================"
echo "Step 2: Run 20-rep negotiation evaluations"
echo "============================================"

# LA-GRPO self-only-v2
echo ""
echo ">>> Evaluating: lagrpo-self-only-v2 (checkpoint-2000)"
python evaluations/run_negotiation_eval.py \
    --checkpoint output/lagrpo-self-only-v2/checkpoint-2000 \
    --repetitions 20 \
    --game-type multi-game \
    --output-dir "$RESULTS_DIR" \
    --lambda-self 1.0 \
    --lambda-welfare 0.0 \
    --lambda-fair 0.0

# Rename the output file to a descriptive name
mv "$RESULTS_DIR/checkpoint-2000.json" "$RESULTS_DIR/lagrpo_self_only_v2_2000.json"
echo "[DONE] lagrpo-self-only-v2 saved to $RESULTS_DIR/lagrpo_self_only_v2_2000.json"

# LA-GRPO all-equal
echo ""
echo ">>> Evaluating: lagrpo-multigame-all-equal (checkpoint-2000)"
python evaluations/run_negotiation_eval.py \
    --checkpoint output/lagrpo-multigame-all-equal/checkpoint-2000 \
    --repetitions 20 \
    --game-type multi-game \
    --output-dir "$RESULTS_DIR" \
    --lambda-self 0.33 \
    --lambda-welfare 0.33 \
    --lambda-fair 0.33

# Rename
mv "$RESULTS_DIR/checkpoint-2000.json" "$RESULTS_DIR/lagrpo_all_equal_2000.json"
echo "[DONE] lagrpo-multigame-all-equal saved to $RESULTS_DIR/lagrpo_all_equal_2000.json"

# LA-GRPO fair-only
echo ""
echo ">>> Evaluating: lagrpo-fair-only (checkpoint-1340)"
python evaluations/run_negotiation_eval.py \
    --checkpoint output/lagrpo-fair-only/checkpoint-1340 \
    --repetitions 20 \
    --game-type multi-game \
    --output-dir "$RESULTS_DIR" \
    --lambda-self 0.0 \
    --lambda-welfare 0.0 \
    --lambda-fair 1.0

# Rename
mv "$RESULTS_DIR/checkpoint-1340.json" "$RESULTS_DIR/lagrpo_fair_only_1340.json"
echo "[DONE] lagrpo-fair-only saved to $RESULTS_DIR/lagrpo_fair_only_1340.json"

echo ""
echo "============================================"
echo "All LA-GRPO 20-rep evaluations complete!"
echo "Results in: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR"/lagrpo_*.json
