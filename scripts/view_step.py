"""View completions log for a specific training step.

Usage: python scripts/view_step.py <step_number> [--log-path <path>]
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="View completions for a specific training step")
    parser.add_argument("step", type=int, help="Step number to display")
    parser.add_argument("--log-path", default="output/grpo_multiturn/completions_log.jsonl",
                        help="Path to completions_log.jsonl")
    args = parser.parse_args()

    entries = []
    with open(args.log_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["step"] == args.step:
                entries.append(entry)

    if not entries:
        available = set()
        with open(args.log_path, encoding="utf-8") as f:
            for line in f:
                available.add(json.loads(line)["step"])
        steps = sorted(available)
        print(f"No entries for step {args.step}.")
        print(f"Available steps: {steps}")
        return

    print(f"=== Step {args.step} — {len(entries)} completions ===\n")
    for i, e in enumerate(entries):
        print(f"--- Completion {i+1}/{len(entries)} ---")
        print(f"Reward:    {e['reward']:.4f}")
        print(f"Advantage: {e['advantage']:.4f}")
        print(f"\nPrompt:\n{e['prompt']}\n")
        print(f"Completion:\n{e['completion']}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
