# RQ5 — Capability Benchmarks Action Plan

**Question (RQ5):** Does cooperation training (GRPO / LA-GRPO with different λ shapes) degrade general LLM capabilities compared to the base model?

**Output target:** One delta table + grouped bar chart per benchmark, showing capability change vs. `OpenPipe/Qwen3-14B-Instruct` base.

---

## 1. Tooling

- **Harness:** [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI).
  - Supports MMLU-Pro, IFEval, GLUE natively.
  - Loads PEFT LoRA adapters via `peft=<path>` model_arg — no merge required.
  - vLLM backend for speed, HF backend as fallback if LoRA + vLLM has issues.

```bash
pip install "lm-eval[vllm]"
```

---

## 2. Benchmarks

| Benchmark | Tests | lm-eval task name | Notes |
|---|---|---|---|
| **MMLU-Pro** | Knowledge + reasoning (14 domains) | `mmlu_pro` | Harder, more discriminative than MMLU |
| **IFEval** | Instruction-following (format/constraint compliance) | `ifeval` | Needs `--apply_chat_template` |
| **GSM8K** | Multi-step math reasoning | `gsm8k` | Good canary for reasoning regression |
| **TruthfulQA** | Hallucination / truthfulness | `truthfulqa_mc2` | Optional, if time |
| ~~GLUE~~ | Classic NLU | `glue` | **Dropped** — dated, not informative for 14B instruct models |

Core suite: **MMLU-Pro + IFEval + GSM8K**. Add TruthfulQA if time allows.

---

## 3. Model Matrix (8 models)

Canonical list of repos, steps, and run_ids lives in [`checkpoints.md`](./checkpoints.md) — that file is the source of truth consumed by `run_capabilities_eval.sh` and `aggregate_capabilities.py`. Update it there, not here.

Grid: GRPO × {self, welfare, fair, equal} + LA-GRPO × {self, fair, equal} + base. `grpo_welfare` has no LA-GRPO counterpart yet, so the grid is asymmetric. All models are directly comparable with the negotiation 20-rep eval.

---

## 4. Command Template

```bash
# LoRA adapter (all trained runs)
lm_eval --model hf \
  --model_args "pretrained=OpenPipe/Qwen3-14B-Instruct,peft=output/<run_name>/checkpoint-<N>,load_in_4bit=True,dtype=bfloat16" \
  --tasks mmlu_pro,ifeval,gsm8k \
  --batch_size auto \
  --apply_chat_template \
  --output_path evaluations/results/capabilities/<run_id>.json \
  --log_samples

# Base model (no peft arg)
lm_eval --model hf \
  --model_args "pretrained=OpenPipe/Qwen3-14B-Instruct,load_in_4bit=True,dtype=bfloat16" \
  --tasks mmlu_pro,ifeval,gsm8k \
  --batch_size auto \
  --apply_chat_template \
  --output_path evaluations/results/capabilities/base.json \
  --log_samples
```

Wrap in `run_capabilities_eval.sh` with a loop over the 7 models. Seed + batch config identical across runs.

---

## 5. Execution Plan

1. **Install + smoke-test** — run `lm_eval` on base with `--limit 10` per task to confirm chat template + 4-bit loading work.
2. **Baseline pass** — full base-model run on MMLU-Pro, IFEval, GSM8K. This is the reference.
3. **Trained runs** — loop over the 6 adapters, same benchmarks.
4. **Aggregate** — one script `aggregate_capabilities.py` reads all JSONs → single CSV with columns `run, benchmark, metric, value, delta_vs_base`.
5. **Plot** — grouped bar chart per benchmark (7 bars, base as dashed line at 0) → drop into thesis.

Estimated wall time on A100 80GB with vLLM: ~45 min MMLU-Pro + ~10 min IFEval + ~10 min GSM8K per model × 8 models ≈ **8–9 h** (overnight run).

---

## 6. Gotchas

- **`--apply_chat_template` is mandatory** for IFEval on instruct models — without it, IFEval scores look bad for reasons unrelated to RL training.
- **LoRA + vLLM**: if `enable_lora=True` + `lora_modules=...` flakes, fall back to HF backend. vLLM speedup not critical for a one-off sweep.
- **4-bit quantisation**: match the training-time quant config (nf4, double quant, bf16 compute) so the base model is evaluated under the same numerical conditions as the adapter-loaded runs. Otherwise base looks artificially better.
- **Batch size `auto`** can OOM on MMLU-Pro; if so pin to 4 or 8.
- **Seed**: `--seed 42` (or equivalent per-task seeds) for reproducibility.

---

## 7. Deliverables

- `evaluations/results/capabilities/<run_id>.json` — raw lm-eval outputs (7 files)
- `evaluations/results/capabilities/capabilities_summary.csv` — aggregated
- `evaluations/results/capabilities/capabilities_results.md` — delta table + interpretation
- `evaluations/results/capabilities/plots/*.png` — grouped bar charts
- Script: `evaluations/run_capabilities_eval.sh`

---

## 8. Success Criteria

- **Green:** All 6 trained models within ±2 pp of base on each benchmark → cooperation training is capability-preserving.
- **Yellow:** One RL variant regresses 2–5 pp → note as a cost of that λ shape.
- **Red:** >5 pp regression on GSM8K or MMLU-Pro → significant capability loss, needs investigation (e.g., KL penalty too low during training).

This directly answers RQ5 and also adds a secondary signal for RQ1 (LA-GRPO vs GRPO) and RQ2 (λ shape trade-offs) beyond negotiation performance.
