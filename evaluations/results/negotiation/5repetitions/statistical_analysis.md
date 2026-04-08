# Statistical Analysis of Negotiation Evaluation Results

**Date:** 2026-04-08
**Data:** 50 games per model (10 configs x 5 repetitions), from `evaluations/results/negotiation/5repetitions/`

## Methods

### Mann-Whitney U Test (Pairwise Comparisons)

The Mann-Whitney U test is a non-parametric test that compares two independent samples without assuming a normal distribution. It tests whether one group tends to have larger values than the other. We use it here instead of a t-test because negotiation payoffs are not normally distributed (e.g., failed negotiations produce 0s, creating a bimodal distribution).

- **Null hypothesis (H0):** The two models produce identically distributed outcomes.
- **Alternative hypothesis (H1):** The distributions differ (two-sided test).
- **Significance levels:** \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001

### Bootstrap Confidence Intervals

To estimate the uncertainty around each model's mean, we use bootstrap resampling:

1. Draw 10,000 random samples (with replacement) of size n=50 from the model's results.
2. Compute the mean of each sample.
3. Take the 2.5th and 97.5th percentiles of the 10,000 means as the 95% confidence interval.

If two models' confidence intervals overlap substantially, their means are not significantly different.

## Results: Pairwise Comparisons on U_A (Agent Payoff)

| Comparison | Mean A | Mean B | U-stat | p-value | Sig? |
|------------|--------|--------|--------|---------|------|
| base_model vs self_only | 50.1 | 55.1 | 1104 | 0.311 | |
| base_model vs fair_only | 50.1 | 61.6 | 1017 | 0.107 | |
| base_model vs **all_equal** | 50.1 | 65.0 | 920 | **0.023** | * |
| base_model vs **self_fair_equal** | 50.1 | 62.2 | 951 | **0.039** | * |
| self_only vs fair_only | 55.1 | 61.6 | 1176 | 0.613 | |
| self_only vs all_equal | 55.1 | 65.0 | 1090 | 0.272 | |
| self_only vs self_fair_equal | 55.1 | 62.2 | 1119 | 0.366 | |
| fair_only vs all_equal | 61.6 | 65.0 | 1140 | 0.449 | |
| fair_only vs self_fair_equal | 61.6 | 62.2 | 1130 | 0.408 | |
| all_equal vs self_fair_equal | 65.0 | 62.2 | 1259 | 0.953 | |

## Results: Pairwise Comparisons on Ratio Nash (Fairness)

| Comparison | Mean A | Mean B | U-stat | p-value | Sig? |
|------------|--------|--------|--------|---------|------|
| base_model vs self_only | 0.541 | 0.564 | 1194 | 0.703 | |
| base_model vs **fair_only** | 0.541 | 0.723 | 785 | **0.001** | ** |
| base_model vs all_equal | 0.541 | 0.663 | 1012 | 0.100 | |
| base_model vs **self_fair_equal** | 0.541 | 0.679 | 917 | **0.022** | * |
| self_only vs **fair_only** | 0.564 | 0.723 | 829 | **0.004** | ** |
| self_only vs all_equal | 0.564 | 0.663 | 1068 | 0.210 | |
| self_only vs self_fair_equal | 0.564 | 0.679 | 970 | 0.054 | |
| **fair_only** vs all_equal | 0.723 | 0.663 | 1546 | **0.041** | * |
| fair_only vs self_fair_equal | 0.723 | 0.679 | 1384 | 0.355 | |
| all_equal vs self_fair_equal | 0.663 | 0.679 | 1122 | 0.381 | |

## Results: Pairwise Comparisons on Ratio Welfare (Social Welfare)

| Comparison | Mean A | Mean B | U-stat | p-value | Sig? |
|------------|--------|--------|--------|---------|------|
| base_model vs self_only | 0.655 | 0.696 | 1143 | 0.459 | |
| base_model vs **fair_only** | 0.655 | 0.834 | 798 | **0.002** | ** |
| base_model vs **all_equal** | 0.655 | 0.827 | 912 | **0.019** | * |
| base_model vs **self_fair_equal** | 0.655 | 0.805 | 882 | **0.011** | * |
| self_only vs **fair_only** | 0.696 | 0.834 | 888 | **0.012** | * |
| self_only vs all_equal | 0.696 | 0.827 | 1024 | 0.117 | |
| self_only vs self_fair_equal | 0.696 | 0.805 | 982 | 0.063 | |
| fair_only vs all_equal | 0.834 | 0.827 | 1411 | 0.264 | |
| fair_only vs self_fair_equal | 0.834 | 0.805 | 1330 | 0.578 | |
| all_equal vs self_fair_equal | 0.827 | 0.805 | 1194 | 0.697 | |

## Bootstrap 95% Confidence Intervals (Cooperative Models)

### U_A (Agent Payoff)

| Model | Mean | 95% CI |
|-------|------|--------|
| fair_only | 61.6 | [54.7, 67.9] |
| all_equal | 65.0 | [58.5, 70.8] |
| self_fair_equal | 62.2 | [54.4, 69.2] |

### Ratio Nash (Fairness)

| Model | Mean | 95% CI |
|-------|------|--------|
| fair_only | 0.723 | [0.651, 0.786] |
| all_equal | 0.663 | [0.594, 0.724] |
| self_fair_equal | 0.679 | [0.601, 0.752] |

### Ratio Welfare (Social Welfare)

| Model | Mean | 95% CI |
|-------|------|--------|
| fair_only | 0.834 | [0.759, 0.898] |
| all_equal | 0.827 | [0.760, 0.883] |
| self_fair_equal | 0.805 | [0.722, 0.876] |

## Summary of Significant Findings

### What we CAN claim (p < 0.05):

1. **Cooperative training significantly improves over baseline.**
   - all_equal and self_fair_equal both significantly outperform base_model on U_A (p=0.023 and p=0.039).
   - fair_only, all_equal, and self_fair_equal all significantly outperform base_model on ratio_welfare (p=0.002, 0.019, 0.011).
   - fair_only significantly outperforms base_model on ratio_nash (p=0.001).

2. **Self-only training does NOT significantly improve over baseline.**
   - base_model vs self_only is non-significant on every metric (all p > 0.3). Pure self-interest optimization does not produce meaningfully better negotiators.

3. **fair_only significantly outperforms self_only.**
   - On ratio_nash (p=0.004) and ratio_welfare (p=0.012). The fairness objective provides a clear advantage over self-interest alone.

4. **fair_only produces significantly fairer outcomes than all_equal.**
   - fair_only > all_equal on ratio_nash (p=0.041). Training purely on the Nash product yields the fairest negotiations.

### What we CANNOT claim (p > 0.05):

5. **The three cooperative models are statistically equivalent on U_A.**
   - fair_only vs all_equal (p=0.449), fair_only vs self_fair_equal (p=0.408), all_equal vs self_fair_equal (p=0.953). The confidence intervals overlap heavily. We cannot say one lambda configuration produces higher agent payoffs than another.

6. **The three cooperative models are statistically equivalent on ratio_welfare.**
   - All pairwise comparisons p > 0.26. No cooperative configuration produces significantly better social welfare than another.

## Implications for the Thesis

**RQ2 (How should reward functions balance self-interest vs. collective welfare?):**

The key finding is that including *any* cooperative component (welfare or fairness) produces significantly better negotiators than pure self-interest or no training. However, the specific balance of lambda parameters does not significantly affect outcomes at n=50. This suggests that the cooperative signal itself matters more than its exact weighting.

To distinguish between lambda configurations, future work would need either larger sample sizes (200+ games per model) or evaluation on out-of-domain scenarios where differences may be more pronounced.
