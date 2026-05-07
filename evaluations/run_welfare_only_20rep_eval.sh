#!/bin/bash
# 20-rep multi-game eval for GRPO welfare-only @ checkpoint-560.
# Mirrors run_v2_20rep_eval.sh but only for the welfare-only model.
set -e

cd /workspace/multiturn-llm-training

OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/negotiation/v2"
mkdir -p "$RESULTS_DIR"

REPS=20
GAME_TYPE="multi-game"

REPO="grpo-multigame-welfare-only"
CKPT="checkpoint-560"
TARGET_DIR="${OUTPUT_DIR}/${REPO}/${CKPT}"

echo "============================================"
echo "Step 1: Download checkpoint"
echo "============================================"
if [ -f "$TARGET_DIR/adapter_model.safetensors" ]; then
    echo "[SKIP] ${REPO}/${CKPT} already exists"
else
    echo "[DOWNLOAD] migub/${REPO} -> ${CKPT}"
    hf download "migub/${REPO}" --include "${CKPT}/*" --local-dir "${OUTPUT_DIR}/${REPO}"
    echo "[OK] ${REPO}/${CKPT}"
fi

echo ""
echo "============================================"
echo "Step 2: Run 20-rep multi-game eval"
echo "============================================"

OUT_NAME="grpo_welfare_only_560"
CKPT_PATH="${TARGET_DIR}"

echo ">>> Evaluating: ${OUT_NAME} (${CKPT_PATH})"
python evaluations/run_negotiation_eval.py \
    --checkpoint "${CKPT_PATH}" \
    --repetitions ${REPS} \
    --game-type ${GAME_TYPE} \
    --output-dir "$RESULTS_DIR" \
    --lambda-self 0.0 \
    --lambda-welfare 1.0 \
    --lambda-fair 0.0

BASE=$(basename "${CKPT_PATH%/}")
if [ -f "$RESULTS_DIR/${BASE}.json" ] && [ "$BASE" != "$OUT_NAME" ]; then
    mv "$RESULTS_DIR/${BASE}.json" "$RESULTS_DIR/${OUT_NAME}.json"
fi
echo "[DONE] ${OUT_NAME} -> $RESULTS_DIR/${OUT_NAME}.json"

echo ""
echo "============================================"
echo "Welfare-only 20-rep eval complete!"
echo "============================================"
ls -la "$RESULTS_DIR/${OUT_NAME}.json" 2>/dev/null
