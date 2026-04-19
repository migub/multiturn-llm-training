# Checkpoint Registry — Capability Evaluation

Source of truth for HuggingFace repo IDs and checkpoint steps used in the RQ5 capability benchmarks. All three consumers must stay in sync with this table:

- `evaluations/run_capabilities_eval.sh` (download + eval)
- `evaluations/scripts/aggregate_capabilities.py` (output keys)
- `evaluations/results/capabilities/action_plan.md` (model matrix)

## Models

| run_id | Method | λ_self | λ_welfare | λ_fair | HF repo | Step |
|---|---|---|---|---|---|---|
| `base` | — | — | — | — | `OpenPipe/Qwen3-14B-Instruct` | — |
| `grpo_self` | GRPO | 1.0 | 0.0 | 0.0 | [`migub/grpo-multigame-self-only`](https://huggingface.co/migub/grpo-multigame-self-only) | 560 |
| `grpo_fair` | GRPO | 0.0 | 0.0 | 1.0 | [`migub/grpo-multigame-fair-only`](https://huggingface.co/migub/grpo-multigame-fair-only) | 560 |
| `grpo_equal` | GRPO | 1/3 | 1/3 | 1/3 | [`migub/grpo-multigame-all-equal`](https://huggingface.co/migub/grpo-multigame-all-equal) | 620 |
| `lagrpo_self` | LA-GRPO | 1.0 | 0.0 | 0.0 | [`migub/lagrpo-self-only-v2`](https://huggingface.co/migub/lagrpo-self-only-v2) | 2000 |
| `lagrpo_fair` | LA-GRPO | 0.0 | 0.0 | 1.0 | [`migub/lagrpo-fair-only`](https://huggingface.co/migub/lagrpo-fair-only) | 1340 |
| `lagrpo_equal` | LA-GRPO | 1/3 | 1/3 | 1/3 | [`migub/lagrpo-multigame-all-equal`](https://huggingface.co/migub/lagrpo-multigame-all-equal) | 2000 |

## Local paths after download

```
output/grpo-multigame-self-only/checkpoint-560/
output/grpo-multigame-fair-only/checkpoint-560/
output/grpo-multigame-all-equal/checkpoint-620/
output/lagrpo-self-only-v2/checkpoint-2000/
output/lagrpo-fair-only/checkpoint-1340/
output/lagrpo-multigame-all-equal/checkpoint-2000/
```

## Machine-readable format (for `run_capabilities_eval.sh`)

Pipe-delimited: `run_id|hf_repo|step`

```
grpo_self|migub/grpo-multigame-self-only|560
grpo_fair|migub/grpo-multigame-fair-only|560
grpo_equal|migub/grpo-multigame-all-equal|620
lagrpo_self|migub/lagrpo-self-only-v2|2000
lagrpo_fair|migub/lagrpo-fair-only|1340
lagrpo_equal|migub/lagrpo-multigame-all-equal|2000
```

## Notes

- Step numbers were selected during negotiation 20-rep evaluation; they are the checkpoints compared in `lagrpo_eval_results.md` and are the same ones used for RQ5 so capability and task-performance deltas are directly comparable.
- `grpo-multigame-self-fair-equal` (step 820) exists but is a separate 4-λ variant not included in the 2×3 grid; add later if needed.
- All adapters are LoRA (r=8, α=16, dropout=0.1, target_modules=all-linear) trained on top of `OpenPipe/Qwen3-14B-Instruct` with the same 4-bit quant config.
