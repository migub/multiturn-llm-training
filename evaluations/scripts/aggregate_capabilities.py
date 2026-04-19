"""Aggregate lm-evaluation-harness JSON outputs into a single CSV.

Reads every `<run_id>.json` in --results-dir, extracts per-task metrics,
computes deltas vs. `base.json`, and writes one flat CSV.
"""

import argparse
import csv
import json
from pathlib import Path

# (task_name, metric_key_in_lm_eval_results)
# lm-eval stores metrics as e.g. "acc,none" or "exact_match,strict-match".
# We pick the primary headline metric per task.
TASK_METRICS = {
    "mmlu_pro": "exact_match,custom-extract",
    "ifeval": "prompt_level_strict_acc,none",
    "gsm8k": "exact_match,strict-match",
}

# Fallback keys if the primary isn't present (harness versions differ).
FALLBACK_METRICS = {
    "mmlu_pro": ["acc,none", "exact_match,none"],
    "ifeval": ["inst_level_strict_acc,none", "prompt_level_loose_acc,none"],
    "gsm8k": ["exact_match,flexible-extract", "acc,none"],
}


def load_results(path: Path) -> dict[str, float]:
    """Return {task: score} from one lm-eval JSON."""
    with path.open() as f:
        data = json.load(f)
    results = data.get("results", {})
    out = {}
    for task, primary in TASK_METRICS.items():
        if task not in results:
            continue
        row = results[task]
        if primary in row:
            out[task] = float(row[primary])
            continue
        for fb in FALLBACK_METRICS.get(task, []):
            if fb in row:
                out[task] = float(row[fb])
                break
        else:
            print(f"[WARN] no known metric for {task} in {path.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--base-run", default="base", help="run_id to use as reference")
    args = ap.parse_args()

    json_files = sorted(args.results_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.results_dir}")

    all_results: dict[str, dict[str, float]] = {}
    for jf in json_files:
        run_id = jf.stem
        all_results[run_id] = load_results(jf)

    if args.base_run not in all_results:
        raise SystemExit(f"Base run '{args.base_run}' not found in results.")
    base_scores = all_results[args.base_run]

    rows = []
    for run_id in sorted(all_results):
        scores = all_results[run_id]
        for task, value in scores.items():
            base_val = base_scores.get(task)
            delta = (value - base_val) if base_val is not None else None
            rows.append(
                {
                    "run": run_id,
                    "benchmark": task,
                    "value": round(value, 4),
                    "base_value": round(base_val, 4) if base_val is not None else "",
                    "delta_vs_base": round(delta, 4) if delta is not None else "",
                    "delta_pp": round(delta * 100, 2) if delta is not None else "",
                }
            )

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.output_csv}")
    print("\nDelta vs base (pp):")
    for run_id in sorted(all_results):
        if run_id == args.base_run:
            continue
        line = [run_id]
        for task in TASK_METRICS:
            v = all_results[run_id].get(task)
            b = base_scores.get(task)
            if v is None or b is None:
                line.append(f"{task}=N/A")
            else:
                line.append(f"{task}={(v - b) * 100:+.2f}")
        print("  " + " | ".join(line))


if __name__ == "__main__":
    main()
