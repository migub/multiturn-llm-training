# Capabilities Eval — A100 Handoff

**Date written:** 2026-04-19
**Context:** RQ5 capability benchmarks (MMLU-Pro, IFEval, GSM8K) were attempted on an RTX 5090 (Blackwell, sm_120). After multiple failed speedup attempts and a blocking vLLM incompatibility, the decision is to migrate this eval to A100 hardware. This document captures everything the next session needs.

---

## TL;DR

- On A100, use **`evaluations/run_capabilities_eval_a100.sh`** (pre-written alongside this doc).
- Models evaluated: `base`, `grpo_equal` (step 620), `lagrpo_equal` (step 2000) — the 2 blended-reward best-overall models, plus base.
- Stack: vLLM backend, bf16 (no 4-bit), 5-shot MMLU-Pro (published-comparable), no `--limit` (full benchmark).
- Expected runtime: **~3-6 h for all 3 models** (vs. ~2+ days on 5090 with HF backend).

---

## What is being evaluated

Three tasks, one `lm_eval` call per model:

| Task | Purpose | Items |
|---|---|---|
| `mmlu_pro` | General knowledge + reasoning, 14 subjects, 5-shot CoT | 12,032 |
| `ifeval` | Instruction-following (structured instructions), 0-shot | 541 |
| `gsm8k` | Grade-school math word problems, 5-shot CoT | 1,319 |

Total ~13,892 items per model × 3 models.

**Checkpoint registry** is `evaluations/results/capabilities/checkpoints.md`. Currently only `grpo_equal` and `lagrpo_equal` are selected via `SELECTED_RUNS` in the shell script — change that list if you want more.

## What was tried on the 5090 (2026-04-19)

Observed throughput per config (same hardware, same model, same tasks):

| Config | Per-item | Projected 3-model total |
|---|---|---|
| HF backend, 4-bit bnb, 5-shot, `batch_size=4`, `max_gen_toks=2048` | **8 s/it** | ~50 h, ~2 days |
| HF, 4-bit, 5-shot, `batch_size=auto`, `max_gen_toks=512`, `--limit 500` | 13 s/it | ~3 days |
| HF, 4-bit, 0-shot, `batch_size=auto`, `max_gen_toks=512`, `--limit 500` | **22 s/it (worse)** | ~6 days |

### Why HF + 5090 was slow

- Qwen3-14B in 4-bit bnb dequantizes on every forward pass — no batched kernels
- GPU utilization stayed at 17-18% — batching-limited, not compute-limited
- 5-shot MMLU-Pro context (~4k tokens) blows up KV cache, so `batch_size=auto` picks 1-2
- 0-shot was counter-intuitively *slower*: with no format template the model generates to `max_gen_toks` rather than stopping early on `"Question:"` — lost more on generation time than we gained on prefill

### Why vLLM failed on 5090

Attempted vLLM 0.19.1 install (successful, 0 torch/transformers changes that mattered). Engine initialization crashed in `torch.ops._vllm_fa2_C.varlen_fwd` with:

```
CUDA error: the provided PTX was compiled with an unsupported toolchain
(cudaErrorUnsupportedPtxVersion)
```

- The precompiled Flash Attention 2 extension in vLLM's wheel has PTX the 5090 driver (570.153, CUDA 12.8) can't load for sm_120 (Blackwell).
- `VLLM_ATTENTION_BACKEND=FLASHINFER` / `TRITON_ATTN` were ignored — engine still selected `FLASH_ATTN`.
- `enforce_eager=True` didn't help — the memory-profiling dummy forward pass still hits FA2.
- This is a bleeding-edge hardware/ecosystem gap, not a bug in our code. Blackwell precompiled kernel support in vLLM is incomplete as of 0.19.1.

## Why A100 works

- A100 = Ampere, sm_80 — ancient in ML-hardware terms, all precompiled vLLM kernels work fine.
- A100 80GB fits Qwen3-14B bf16 (~28GB weights) with ~45GB+ left for KV cache and batching — no need for 4-bit quant.
- Paged attention means the 5-shot MMLU-Pro context is no longer a batching killer.
- Expected effective batch 16-32 instead of 1-2 → ~5-10x speedup over HF baseline.

## A100 setup checklist (do this first on the A100)

1. Clone the repo (or `git pull`) and `cd multiturn-llm-training`.
2. Install vLLM — it should "just work" on sm_80:
   ```
   pip install vllm
   ```
   (No FlashInfer or backend env vars needed.)
3. Ensure the 2 adapters are downloaded locally. Either re-run step 1 of the main eval script, or:
   ```
   hf download migub/grpo-multigame-all-equal --include "checkpoint-620/*" --local-dir output/grpo-multigame-all-equal
   hf download migub/lagrpo-multigame-all-equal --include "checkpoint-2000/*" --local-dir output/lagrpo-multigame-all-equal
   ```
4. Run the A100 eval:
   ```
   tmux new -s rq5-capabilities
   bash evaluations/run_capabilities_eval_a100.sh 2>&1 | tee evaluations/results/capabilities/run.log
   ```
5. Detach with `Ctrl+b d`. Expect 3-6 hours end-to-end.

## LoRA adapter strategy

The A100 script uses **pre-merging**: for each adapter, load base + adapter, `merge_and_unload()`, save as a full model, then point vLLM at the merged model directory. This is the simplest and most compatible approach.

- Pros: standard vLLM loading, no special flags, same as loading any HF model
- Cons: ~28 GB of temp disk per adapter (2 adapters → 56 GB peak)
- Merged models are kept across runs (cached at `output/merged/<run_id>/`) so a rerun doesn't re-merge

The alternative — vLLM's native `enable_lora` + `lora_local_path` via lm_eval — is faster to set up but has compatibility rough edges. If merging is a pain, try:
```
--model vllm --model_args pretrained=<base>,enable_lora=True,max_lora_rank=8,lora_local_path=<adapter_path>,dtype=bfloat16
```

## Known issues to watch for

- **vLLM + chat template**: lm_eval passes `--apply_chat_template` which works with vLLM, but double-check the first few outputs look sensible (no raw `<|im_start|>` leaking).
- **GSM8K stop tokens**: Qwen3's chat template may emit `<|im_end|>` before `Question:` — existing config already lists both in `until`.
- **`--limit` warning from lm_eval**: the 5090 script set `--limit 500` for dev speed; the A100 script does NOT use `--limit` (full benchmark). If you still want a quick dev pass, add it back.

## Files touched / relevant

- `evaluations/run_capabilities_eval.sh` — the 5090 HF-backend script. Currently filters to `grpo_equal` + `lagrpo_equal`, has `--limit 500 --num_fewshot 0 --gen_kwargs max_gen_toks=512 --batch_size auto`. Keep as-is for reproducibility, or delete — it's been superseded.
- `evaluations/run_capabilities_eval_a100.sh` — NEW, vLLM backend, bf16, 5-shot, full benchmark.
- `evaluations/results/capabilities/checkpoints.md` — source of truth for which adapters get evaluated; both scripts read from this.
- `evaluations/results/capabilities/action_plan.md` — original RQ5 plan.
- `evaluations/results/capabilities/run.log.*-killed` — killed-run logs kept for debugging; safe to delete.

## Handoff state as of 2026-04-19 22:45

- No eval is running. All tmux sessions killed.
- GPU (5090) is idle.
- Local adapters `grpo_equal` (step 620) and `lagrpo_equal` (step 2000) are downloaded at `output/grpo-multigame-all-equal/checkpoint-620/` and `output/lagrpo-multigame-all-equal/checkpoint-2000/`. They do NOT need re-downloading on the 5090, but DO need re-downloading on the A100 (step 3 above).
- vLLM 0.19.1 was installed on the 5090 and is unusable there. On the A100 you can reinstall fresh; nothing from the 5090 environment needs to come with you.
