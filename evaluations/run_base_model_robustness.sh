#!/usr/bin/env bash
# RQ4 robustness eval — base Qwen3-14B (no LoRA) vs 4 adversarial personas.
# Output files renamed to base_model_<persona>.json to match ROBUSTNESS_EVAL.md naming.
set -euo pipefail

cd /workspace/multiturn-llm-training

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
        --checkpoint none \
        --output-dir "$RESULTS_DIR"
    if [ -f "$RESULTS_DIR/none_${persona}.json" ]; then
        mv "$RESULTS_DIR/none_${persona}.json" \
           "$RESULTS_DIR/base_model_${persona}.json"
        echo "[RENAMED] -> base_model_${persona}.json"
    fi
done

echo "================================================================"
echo "DONE  ($(date))"
echo "================================================================"
ls -la "$RESULTS_DIR"/base_model_*.json
