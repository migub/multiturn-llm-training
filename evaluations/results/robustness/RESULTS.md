# RQ4 Robustness Eval — Results

Adversarial personas: `hardball`, `deceptive`, `anchoring`, `stubborn`.
Setup: 14 games × 10 reps × 5 max rounds. Lambdas: `λ_self=0, λ_welfare=0, λ_fair=1` (matches LA-GRPO fair-only training).

## Base model (Qwen3-14B, no LoRA)

| persona   | agree | U_A  | U_B  | ratio_self | ratio_rcoop | agreed_rcoop |
|-----------|-------|------|------|------------|-------------|--------------|
| hardball  | 26.4% | 14.5 | 19.8 | 0.145      | 0.145       | 0.550        |
| deceptive | 65.7% | 39.8 | 40.9 | 0.398      | 0.398       | 0.605        |
| anchoring | 55.0% | 36.5 | 33.9 | 0.365      | 0.365       | 0.663        |
| stubborn  | 22.1% | 16.2 | 14.4 | 0.162      | 0.162       | 0.732        |

## LA-GRPO fair-only (checkpoint-1340)

| persona   | agree | U_A  | U_B  | ratio_self | ratio_rcoop | agreed_rcoop |
|-----------|-------|------|------|------------|-------------|--------------|
| hardball  | 56.4% | 30.2 | 48.4 | 0.302      | 0.367       | 0.651        |
| deceptive | 81.4% | 47.6 | 52.3 | 0.476      | 0.606       | 0.744        |
| anchoring | 77.1% | 44.0 | 52.0 | 0.440      | 0.515       | 0.667        |
| stubborn  | 14.3% | 11.1 |  9.5 | 0.111      | 0.120       | 0.843        |

## Exploitation gap (Δ = lagrpo_fair_only − base)

| persona   | Δ agree   | Δ U_A    | Δ ratio_rcoop |
|-----------|-----------|----------|---------------|
| hardball  | +30.0 pp  | +15.7    | +0.222        |
| deceptive | +15.7 pp  |  +7.8    | +0.208        |
| anchoring | +22.1 pp  |  +7.5    | +0.150        |
| stubborn  |  −7.8 pp  |  −5.1    | −0.042        |

## Headline

Fair-only training **helps against hardball, deceptive, and anchoring** — more agreements and higher payoffs across the board. The hardball gap is largest: base agent gets squeezed (U_A=14.5), trained agent recovers to U_A=30.2 — so the cooperative agent isn't *more* exploitable than baseline, it's still imperfect.

Against **stubborn**, the trained agent slightly underperforms the base model: both collapse (agreement <25%), but the trained agent is marginally less able to break the deadlock. When agreement *does* happen, however, quality is high (`agreed_rcoop = 0.843` — the best of any cell).

## Files

- `base_model_{hardball,deceptive,anchoring,stubborn}.json`
- `lagrpo_fair_only_1340_{hardball,deceptive,anchoring,stubborn}.json`

Cooperative-baseline (non-adversarial opponent) lives in the 20-rep eval directory: `evaluations/results/.../lagrpo_fair_only_1340.json`.
