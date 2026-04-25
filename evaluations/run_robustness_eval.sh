#!/usr/bin/env bash
# RQ4 robustness eval — T2: prompt-injected adversarial opponents.
#
# Loops over {hardball, deceptive, anchoring, stubborn} opponent personas and
# evaluates each checkpoint against them. The default 'cooperative' selfplay
# baseline should already exist from your prior 20-rep eval; re-run it below
# if you want fresh numbers with identical seeds.
#
# Each (checkpoint, persona) combination is one pass of run_negotiation_eval.py.
# Output goes to evaluations/results/robustness/ so it doesn't overwrite the
# selfplay results in evaluations/results/negotiation/.
#
# Usage:
#   bash evaluations/run_robustness_eval.sh                    # full matrix
#   bash evaluations/run_robustness_eval.sh hardball deceptive # subset
#   REPETITIONS=5 bash evaluations/run_robustness_eval.sh      # smaller run

set -euo pipefail

REPETITIONS="${REPETITIONS:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
NUM_GAMES="${NUM_GAMES:-14}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/results/robustness}"

# Personas to sweep (positional args override defaults)
if [ "$#" -gt 0 ]; then
    PERSONAS=("$@")
else
    PERSONAS=(hardball deceptive anchoring stubborn)
fi

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "RQ4 Robustness Eval — T2 (prompt-injected adversaries)"
echo "  Personas:    ${PERSONAS[*]}"
echo "  Repetitions: $REPETITIONS"
echo "  Max rounds:  $MAX_ROUNDS"
echo "  Num games:   $NUM_GAMES"
echo "  Output dir:  $OUTPUT_DIR"
echo "================================================================"

for persona in "${PERSONAS[@]}"; do
    echo ""
    echo "---------------- Persona: $persona ----------------"
    python evaluations/run_negotiation_eval.py \
        --opponent-persona "$persona" \
        --repetitions "$REPETITIONS" \
        --max-rounds "$MAX_ROUNDS" \
        --num-games "$NUM_GAMES" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "Done. Results under $OUTPUT_DIR/"
