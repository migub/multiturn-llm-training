# Capabilities Eval — RQ5 Summary

**Run date:** 2026-04-19 → 2026-04-20 (overnight, A100 80GB)
**Base model:** `OpenPipe/Qwen3-14B-Instruct`
**Benchmarks:** MMLU-Pro (5-shot), IFEval, GSM8K
**Backend:** lm-evaluation-harness 0.4.11 + vLLM 0.19.1, `bfloat16`, `max_model_len=8192`, `gpu_memory_utilization=0.9`

## Results

| Run | MMLU-Pro | IFEval (prompt-strict) | GSM8K (strict-match) |
|---|---|---|---|
| base Qwen3-14B-Instruct | 0.6903 | 0.8521 | 0.8802 |
| grpo_equal (ckpt 620)   | 0.6902 | 0.8577 | 0.8817 |
| lagrpo_equal (ckpt 2000)| 0.6876 | 0.8484 | 0.8795 |

### Delta vs. base (percentage points)

| Run | MMLU-Pro | IFEval | GSM8K |
|---|---|---|---|
| grpo_equal   | −0.02 | **+0.55** | **+0.15** |
| lagrpo_equal | −0.27 | −0.37 | −0.08 |

## Interpretation (RQ5)

> *"Does cooperation training affect general LLM capabilities?"*

All absolute deltas sit **within ±0.6 pp** — essentially noise-level on benchmarks this size. Cooperation/Pareto-efficiency training via GRPO and LA-GRPO on the equal-weighted reward (λ_self = λ_welfare = λ_fair = 1) **does not measurably degrade general capabilities** on MMLU-Pro, IFEval, or GSM8K.

- `grpo_equal` nudged IFEval and GSM8K up slightly and held MMLU-Pro flat.
- `lagrpo_equal` regressed by <0.4 pp on all three — within run-to-run noise for single-seed evals.

**Takeaway for the thesis:** the negotiation-specific RL training preserved the model's generalist capability profile. Supports the claim that cooperative alignment can be trained without "capability tax" on the eval suite tested.

## Checkpoints evaluated

- `migub/grpo-multigame-all-equal` @ checkpoint-620
- `migub/lagrpo-multigame-all-equal` @ checkpoint-2000

Selection rule lives in `evaluations/results/capabilities/checkpoints.md`; script reads it via `SELECTED_RUNS="grpo_equal lagrpo_equal"` in `evaluations/run_capabilities_eval_a100.sh`.

## Artifacts in this folder

- `capabilities_summary.csv` — the flat 9-row table (3 runs × 3 benchmarks)
- `base_<timestamp>.json`, `grpo_equal_<timestamp>.json`, `lagrpo_equal_<timestamp>.json` — full lm_eval outputs with per-subtask scores, versions, and reproducibility metadata
- `samples_<task>_<timestamp>.jsonl` — per-sample logs (prompt, response, gold, correct) from `--log_samples`
- `run.log` — full script stdout/stderr
- `run.log.prev*` — logs from pre-fix attempts (kept for debugging, safe to delete)
- `action_plan.md` — original action plan for RQ5
- `checkpoints.md` — checkpoint registry

## Run log / fixes applied during the run

The handoff script (`evaluations/run_capabilities_eval_a100.sh`) had to be adapted on the fly; pod environment was missing several runtime deps and storage assumptions didn't hold:

1. **mfs disk quota (~20 GB) hit during LoRA merge.** The first merge was writing a single 19.5 GB `model.safetensors` directly into the 20 GB-quota mfs volume. Relocated `MERGED_DIR` from `output/merged` (mfs) to `/root/merged` (overlay SSD, 56 GB free, also ~1000× faster).
2. **Restructured Step 2 → Step 3 to interleave merge + eval + cleanup** so only one merged model (~28 GB) is on disk at a time. Without this, two merged models would not both fit on the overlay.
3. **Moved merge from CPU to GPU** (`device_map="cuda"`): ~30 s per adapter instead of ~10 min on CPU.
4. **Added `max_shard_size="5GB"`** to `save_pretrained` for cleaner shard writes (cosmetic).
5. **Missing pip deps** installed on the fly: `ray` (lm_eval's vLLM backend), `langdetect`, `immutabledict`, `nltk` (+ `punkt` / `punkt_tab` data — IFEval needs all three), `sentencepiece`, `antlr4-python3-runtime`, `hf_transfer`.
6. **Patched `evaluations/scripts/aggregate_capabilities.py`** to strip lm_eval's `_YYYY-MM-DDTHH-MM-SS.microseconds` suffix from JSON stems (new default lm_eval behavior when `--log_samples` is on). Without this, the aggregate step would not have matched `base.json` as the reference run and would have failed.

## Reproducibility

```bash
# Full command used:
bash evaluations/run_capabilities_eval_a100.sh
# Key knobs (hardcoded in script):
#   TASKS=mmlu_pro,ifeval,gsm8k
#   SELECTED_RUNS="grpo_equal lagrpo_equal"
#   VLLM_ARGS_BASE="dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=8192,tensor_parallel_size=1"
#   COMMON_ARGS="--tasks $TASKS --batch_size auto --apply_chat_template --log_samples --seed 42"
```

Total wall time: ~3 h 20 min on A100 80 GB (base ~50 min, grpo_equal ~50 min, lagrpo_equal ~50 min, merges ~30 s each).
