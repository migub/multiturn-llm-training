#!/usr/bin/env bash
# RQ4 robustness eval for LA-GRPO fair-only (checkpoint-1340).
# Loops over 4 adversarial personas, renames outputs to lagrpo_fair_only_1340_<persona>.json.
set -euo pipefail

cd /workspace/multiturn-llm-training

CKPT="output/lagrpo-fair-only/checkpoint-1340"
RESULTS_DIR="evaluations/results/robustness"
PERSONAS=(hardball deceptive anchoring stubborn)

mkdir -p "$RESULTS_DIR"

for persona in "${PERSONAS[@]}"; do
    echo "================================================================"
    echo ">>> Persona: $persona  ($(date))"
    echo "================================================================"
    python evaluations/run_negotiation_eval.py \
        --opponent-persona "$persona" \
        --repetitions 10 --num-games 14 --max-rounds 5 \
        --checkpoint "$CKPT" \
        --lambda-self 0.0 --lambda-welfare 0.0 --lambda-fair 1.0 \
        --output-dir "$RESULTS_DIR"
    if [ -f "$RESULTS_DIR/checkpoint-1340_${persona}.json" ]; then
        mv "$RESULTS_DIR/checkpoint-1340_${persona}.json" \
           "$RESULTS_DIR/lagrpo_fair_only_1340_${persona}.json"
        echo "[RENAMED] -> lagrpo_fair_only_1340_${persona}.json"
    fi
done

echo "================================================================"
echo "DONE  ($(date))"
echo "================================================================"
ls -la "$RESULTS_DIR"/lagrpo_fair_only_1340_*.json
