# Statistical Analysis of Negotiation Evaluation Results (20 Repetitions)

**Date:** 2026-04-09
**Data:** 560 games per model (14 configs x 2 roles x 20 repetitions)

## Methods

### Mann-Whitney U Test (Pairwise Comparisons)

The **Mann-Whitney U test** is a non-parametric test that compares two independent samples without assuming a normal distribution. It tests whether one group tends to have larger values than the other. We use it instead of a t-test because negotiation payoffs are not normally distributed (failed negotiations produce 0s, creating a bimodal distribution).

- **H0:** The two models produce the same distribution of scores
- **H1:** The distributions differ (two-sided test)
- Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001

### Chi-squared Test (Agreement Rate)

For comparing binary agreement rates between models, we use the **chi-squared test of independence** on 2x2 contingency tables.

### Bootstrap Confidence Intervals

We compute 95% confidence intervals using **non-parametric bootstrap** with 10,000 resamples. For each resample, we draw N samples with replacement from the original data and compute the mean. The 2.5th and 97.5th percentiles of the bootstrap distribution give the confidence interval.

## Pairwise Comparisons: U_A

| Comparison | Mean A | Mean B | U-stat | p-value | Sig |
|---|---|---|---|---|---|
| base vs self_only | 49.9 | 54.9 | 139164 | 0.0010 | ** |
| base vs fair_only | 49.9 | 55.0 | 143187 | 0.0115 | * |
| base vs all_equal | 49.9 | 60.3 | 128404 | 0.0000 | *** |
| base vs self_fair_equal | 49.9 | 57.0 | 135476 | 0.0001 | *** |
| self_only vs fair_only | 54.9 | 55.0 | 160636 | 0.4769 | |
| self_only vs all_equal | 54.9 | 60.3 | 148991 | 0.1475 | |
| self_only vs self_fair_equal | 54.9 | 57.0 | 155097 | 0.7517 | |
| fair_only vs all_equal | 55.0 | 60.3 | 142006 | 0.0061 | ** |
| fair_only vs self_fair_equal | 55.0 | 57.0 | 148936 | 0.1448 | |
| all_equal vs self_fair_equal | 60.3 | 57.0 | 162056 | 0.3294 | |

## Pairwise Comparisons: Ratio Nash

| Comparison | Mean A | Mean B | U-stat | p-value | Sig |
|---|---|---|---|---|---|
| base vs self_only | 0.585 | 0.581 | 156576 | 0.9669 | |
| base vs fair_only | 0.585 | 0.664 | 135443 | 0.0001 | *** |
| base vs all_equal | 0.585 | 0.692 | 130286 | 0.0000 | *** |
| base vs self_fair_equal | 0.585 | 0.625 | 144752 | 0.0253 | * |
| self_only vs fair_only | 0.581 | 0.664 | 136419 | 0.0002 | *** |
| self_only vs all_equal | 0.581 | 0.692 | 131722 | 0.0000 | *** |
| self_only vs self_fair_equal | 0.581 | 0.625 | 145964 | 0.0443 | * |
| fair_only vs all_equal | 0.664 | 0.692 | 152828 | 0.4622 | |
| fair_only vs self_fair_equal | 0.664 | 0.625 | 166272 | 0.0792 | |
| **all_equal vs self_fair_equal** | **0.692** | **0.625** | **170254** | **0.0127** | **\*** |

## Pairwise Comparisons: Ratio Welfare

| Comparison | Mean A | Mean B | U-stat | p-value | Sig |
|---|---|---|---|---|---|
| base vs self_only | 0.702 | 0.723 | 147260 | 0.0726 | |
| base vs fair_only | 0.702 | 0.808 | 130718 | 0.0000 | *** |
| base vs all_equal | 0.702 | 0.840 | 126818 | 0.0000 | *** |
| base vs self_fair_equal | 0.702 | 0.757 | 143484 | 0.0124 | * |
| self_only vs fair_only | 0.723 | 0.808 | 141116 | 0.0030 | ** |
| self_only vs all_equal | 0.723 | 0.840 | 137837 | 0.0003 | *** |
| self_only vs self_fair_equal | 0.723 | 0.757 | 153855 | 0.5790 | |
| fair_only vs all_equal | 0.808 | 0.840 | 153792 | 0.5676 | |
| **fair_only vs self_fair_equal** | **0.808** | **0.757** | **169988** | **0.0127** | **\*** |
| **all_equal vs self_fair_equal** | **0.840** | **0.757** | **173044** | **0.0022** | **\*\*** |

## Pairwise Comparisons: Agreement Rate (Chi-squared)

| Comparison | Rate A | Rate B | Chi2 | p-value | Sig |
|---|---|---|---|---|---|
| base vs self_only | 78.6% | 79.8% | 0.20 | 0.6587 | |
| base vs fair_only | 78.6% | 89.6% | 24.85 | 0.0000 | *** |
| base vs all_equal | 78.6% | 92.7% | 44.13 | 0.0000 | *** |
| base vs self_fair_equal | 78.6% | 83.8% | 4.58 | 0.0324 | * |
| self_only vs fair_only | 79.8% | 89.6% | 20.13 | 0.0000 | *** |
| self_only vs all_equal | 79.8% | 92.7% | 37.95 | 0.0000 | *** |
| self_only vs self_fair_equal | 79.8% | 83.8% | 2.64 | 0.1040 | |
| fair_only vs all_equal | 89.6% | 92.7% | 2.84 | 0.0921 | |
| **fair_only vs self_fair_equal** | **89.6%** | **83.8%** | **7.93** | **0.0049** | **\*\*** |
| **all_equal vs self_fair_equal** | **92.7%** | **83.8%** | **20.62** | **0.0000** | **\*\*\*** |

## Bootstrap 95% Confidence Intervals

| Model | U_A Mean [95% CI] | Ratio Nash [95% CI] | Ratio Welfare [95% CI] |
|-------|-------------------|---------------------|------------------------|
| base_model | 49.9 [47.2, 52.6] | 0.585 [0.556, 0.613] | 0.702 [0.671, 0.734] |
| self_only | 54.9 [52.0, 57.7] | 0.581 [0.552, 0.610] | 0.723 [0.691, 0.753] |
| fair_only | 55.0 [52.4, 57.6] | 0.664 [0.637, 0.689] | 0.808 [0.782, 0.833] |
| all_equal | 60.3 [58.0, 62.6] | 0.692 [0.669, 0.716] | 0.840 [0.819, 0.861] |
| self_fair_equal | 57.0 [54.4, 59.6] | 0.625 [0.596, 0.653] | 0.757 [0.727, 0.786] |

## Summary of Significant Findings

### Clearly significant (p < 0.01)
- **All trained models > base** on U_A
- **fair_only, all_equal > base** on ratio_nash, ratio_welfare, agreement rate (all p < 0.001)
- **fair_only, all_equal > self_only** on ratio_nash, ratio_welfare, agreement rate (all p < 0.01)
- **all_equal > fair_only** on U_A (p = 0.006)
- **all_equal > self_fair_equal** on ratio_welfare (p = 0.002), agreement rate (p < 0.001)

### Marginally significant (p < 0.05)
- **all_equal > self_fair_equal** on ratio_nash (p = 0.013)
- **fair_only > self_fair_equal** on agreement rate (p = 0.005), ratio_welfare (p = 0.013)

### Not significant
- self_only vs fair_only on U_A (p = 0.48)
- self_only vs self_fair_equal on U_A (p = 0.75)
- all_equal vs self_fair_equal on U_A (p = 0.33)
- fair_only vs all_equal on ratio_nash (p = 0.46), ratio_welfare (p = 0.57), agreement rate (p = 0.09)

### Model ranking (by statistical groupings)

**Tier 1:** all_equal — significantly best on U_A (vs fair_only), ratio_welfare (vs self_fair_equal), agreement rate (vs self_fair_equal)

**Tier 2:** fair_only, self_fair_equal — both significantly better than base on most metrics; fair_only significantly better than self_fair_equal on agreement rate and ratio_welfare

**Tier 3:** self_only, base_model — self_only improves U_A over base but not ratio_nash or agreement rate; effectively no meaningful difference on cooperative metrics
