Step 1: Per-sample payoffs (lines 469-477) - The evaluator (GPT-4o-mini) reads the negotiation dialogue and extracts what was agreed - lookup_payoff() maps the agreed values to payoff points → gives U_A and U_B  
 Step 2: Per-sample welfare calculation (line 479)  
 social_welfare = U_A + U_B Step 3: Normalize against the maximum possible (line 490) ratio_welfare = social_welfare / max_metrics["max_social_welfare"]
max_social_welfare comes from compute_max_metrics() which brute-forces all possible outcome combinations (e.g. 11×11 = 121 for 2 issues) and finds the  
 maximum achievable U_A + U_B. This makes the ratio 0-1 and comparable across games.  
 Step 4: Collect into batch (line 498)  
 batch_ratio_welfare.append(ratio_welfare)

Step 5: Average across all samples in the batch (line 533)  
 "negotiation/ratio_welfare_mean": sum(batch_ratio_welfare) / n,  
 Where n = number of samples in the batch (= num_generations).

Step 6: Also per-archetype (lines 548, 557)  
 metrics[f"negotiation/{arch}/ratio_welfare_mean"] = sum(vals["ratio_welfare"]) / m  
 Step 7: Log to wandb (line 561)  
 wandb.log(metrics, commit=False)
