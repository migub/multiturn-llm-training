# Lambda Selection: Literature Review and Recommendations

**Date:** 2026-03-31
**Context:** How to set lambda_self, lambda_welfare, lambda_fair in the cooperative reward R_coop, and why the "all-equal" (1/3, 1/3, 1/3) configuration underperformed.

---

## 1. Why "All-Equal" Is a Weak Choice

The all-equal run (lambda_self=1/3, lambda_welfare=1/3, lambda_fair=1/3) was the worst performer in our three GRPO runs. This is not surprising — equal weighting is a naive default with **no theoretical justification**:

1. **Welfare provides zero signal in distributive games.** In pure distributive games (U_A + U_B = constant), the welfare term has zero variance. ~31% of our games are pure distributive or contain distributive issues. Equal weighting wastes 1/3 of the reward budget on noise for these games.

2. **Nash product already captures efficiency AND fairness.** The Nash product (U_A × U_B) is maximized when both the total value is high AND the split is balanced. Adding welfare on top is partially redundant — welfare only adds unique signal in integrative/compatible games where total value varies but the split is already balanced.

3. **Self-interest acts as a stabilizer.** Without sufficient self-interest weight, the agent can learn degenerate strategies (e.g., always concede to ensure high Nash product). At lambda_self=1/3, this stabilization is too weak.

4. **Linear scalarization's theoretical limitation.** Hayes et al. (2022) establish that linear scalarization can only discover solutions on the **convex hull** of the Pareto front. Points in concave regions are unreachable regardless of weights. However, our formula partially avoids this because the Nash product term (U_A × U_B) is inherently non-linear, helping reach solutions that pure linear combinations of U_A and U_B would miss.

**Our data confirms this:** All-equal achieved the worst agreed-only metrics (ratio_self=0.423, ratio_nash=0.386) — even when it reached deals, they were low quality.

---

## 2. What the Literature Says About Reward Weighting

### 2.1 The Three Classical Social Welfare Functions

Our three reward components correspond to well-established concepts from social choice theory and cooperative game theory:

| Component | Formula | Social Welfare Function | Key Property |
|-----------|---------|------------------------|--------------|
| lambda_self × (U_A / max_U_A) | Self-utility | **Individual rationality** | Ensures the agent has incentive to participate |
| lambda_welfare × ((U_A + U_B) / max_welfare) | Utilitarian welfare | **Utilitarian** (Bentham, 1789) | Maximizes total value created ("size of the pie") |
| lambda_fair × ((U_A × U_B) / max_nash) | Nash product | **Nash bargaining solution** (Nash, 1950) | Maximizes balanced, efficient outcomes |

Nash (1950) proved that the Nash bargaining solution is the **unique** function satisfying four axioms:
- **Pareto efficiency:** No alternative makes one party better off without hurting the other
- **Symmetry:** Identical agents get identical outcomes
- **Scale invariance:** Rescaling one party's utilities doesn't change the solution
- **Independence of irrelevant alternatives:** Removing non-chosen options doesn't change the solution

Social welfare (U_A + U_B) satisfies Pareto efficiency but **not scale invariance** — if you double one player's payoff scale, the utilitarian optimum shifts entirely toward that player.

**Reference:** Nash, J. F. (1950). "The Bargaining Problem." *Econometrica*, 18(2), 155–162.

### 2.2 "The Unreasonable Fairness of Maximum Nash Welfare"

Caragiannis et al. (2019) proved a remarkable result: maximizing Nash welfare (the product of utilities) simultaneously achieves **envy-freeness up to one good** and **Pareto efficiency** in allocation problems. They call this "unreasonable" because Nash welfare was designed for efficiency, yet it also provides strong fairness guarantees "for free."

This is the strongest theoretical argument for giving the Nash product term significant weight in our reward function. It suggests that lambda_fair should be the dominant cooperative term — it captures both efficiency and fairness in a single metric, while welfare only captures efficiency.

**Reference:** Caragiannis, I., Kurokawa, D., Moulin, H., Procaccia, A. D., Shah, N., & Wang, J. (2019). "The Unreasonable Fairness of Maximum Nash Welfare." *ACM Transactions on Economics and Computation*, 7(3), 1–32. (Originally presented at ACM EC 2016.)

### 2.3 Prosocial Reward Shaping: The Peysakhovich & Lerer Finding

Peysakhovich & Lerer (2017) trained deep RL agents with a simple other-regarding reward:

```
R_i = (1 - alpha) × U_self + alpha × U_other
```

where alpha ∈ [0, 1] controls the degree of other-regard. They tested this across randomly generated Stag Hunt games (coordination games with a cooperative equilibrium that Pareto-dominates a selfish equilibrium).

**Key finding:** Agents trained with moderate prosociality (positive alpha, roughly in the range 0.2–0.5) **outperformed purely selfish agents (alpha=0) even when evaluated selfishly at test time**. The prosocial bias during training helped agents escape coordination failures.

This is directly relevant to our setup. Translated to our formula:
- Their `alpha × U_other` term ≈ our `lambda_welfare` term (both reward the other party)
- Their finding suggests **~20–50% other-regard** is the sweet spot
- Below this range, agents defect too often; above it, agents become exploitable

**Caveat:** The specific value alpha≈0.3 often cited is approximate. The paper shows a *range* of positive alpha values outperform alpha=0, not a single optimal point. The exact optimum depends on the game structure.

**Reference:** Peysakhovich, A. & Lerer, A. (2017). "Prosocial Learning Agents Solve Generalized Stag Hunts Better than Selfish Ones." *arXiv:1709.02865*. (Presented at AAMAS 2018 workshop.)

### 2.4 Inequity Aversion (Fehr-Schmidt Model)

Hughes et al. (2018) adapted the Fehr-Schmidt inequity aversion model from behavioral economics into multi-agent RL:

```
U_i(r) = r_i - alpha × max(r_j - r_i, 0) - beta × max(r_i - r_j, 0)
```

Where:
- **alpha** penalizes *disadvantageous* inequity (the other gets more than you)
- **beta** penalizes *advantageous* inequity (you get more than the other)
- Constraint: alpha >= beta >= 0 (people dislike being worse off more)

**Key finding:** Inequity-averse agents learned to cooperate in sequential social dilemmas (Cleanup, Harvest) where standard self-interested agents defect. Both alpha > 0 AND beta > 0 were needed — disliking disadvantageous inequity alone was insufficient.

**Relevance to our work:** Our Nash product term (U_A × U_B) implicitly captures inequity aversion. The product drops sharply when one party gets much more than the other, similar to the Fehr-Schmidt beta penalty. This provides an alternative theoretical justification for the Nash product: it acts as a **smooth, symmetric inequity penalty** without requiring separate alpha/beta parameters.

**Reference:** Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Duéñez-Guzmán, E., et al. (2018). "Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas." *NeurIPS 2018*. arXiv:1803.08884.

### 2.5 Social Diversity Matters More Than Specific Weights

McKee et al. (2020) trained populations of agents with diverse **Social Value Orientation (SVO)**:

```
U = cos(theta) × R_self + sin(theta) × R_other
```

where theta is the SVO angle (0° = selfish, 45° = fully prosocial).

**Key finding:** The **diversity of preferences within a population** matters more than the specific weight values. A heterogeneous population (agents with different thetas) outperformed any homogeneous population, even one tuned to "optimal" weights.

**Implications for our work:**
- The exact lambda values may matter less than we think
- Training against opponents with *varied* lambda values (or varied strategies) could be more robust than a fixed opponent
- This contextualizes why all-equal wasn't dramatically worse — the exact split matters less than having *some* cooperative signal

**Reference:** McKee, K. R., Gemp, I., McWilliams, B., Duéñez-Guzmán, E. A., Hughes, E., & Leibo, J. Z. (2020). "Social Diversity and Social Preferences in Mixed-Motive Reinforcement Learning." *AAMAS 2020*.

### 2.6 Prior Negotiation RL Work Uses Pure Self-Interest

The two foundational LLM negotiation papers both used purely self-interested rewards:

- **Lewis et al. (2017)** — "Deal or No Deal? End-to-End Learning for Negotiation Dialogues." *EMNLP 2017*. Reward = agent's own utility from items received. Zero-sum framing.

- **He et al. (2018)** — "Decoupling Strategy and Generation in Negotiation Dialogues." *EMNLP 2018*. Reward = how favorable the agreed price was. Self-interested.

- **Cao et al. (2018)** — "Emergent Communication through Negotiation." *ICLR 2018*. Also self-interested reward. Nash product was used as an **evaluation metric**, not as a training signal.

**This means our approach of directly incorporating Nash product into the training reward is fairly novel in the negotiation RL literature.** Most prior work maximizes self-interest during training and then evaluates with cooperative metrics post-hoc. Our cooperative reward function trains for cooperation directly — a key contribution of the thesis.

### 2.7 The Cooperative AI Research Agenda

Dafoe et al. (2021) laid out the research agenda for cooperative AI, arguing that the competitive/zero-sum framing dominates current multi-agent RL and that explicit cooperative capabilities are under-researched. They identify five pillars: understanding, communication, commitment, institutions, and norms.

Our work falls under "communication" (negotiation) with cooperative reward shaping. The fact that most prior negotiation RL uses self-interested rewards (Section 2.6) confirms the gap that Dafoe et al. identified.

**Reference:** Dafoe, A., Hughes, E., Bachrach, Y., Collins, T., McKee, K. R., Leibo, J. Z., Larson, K., & Graepel, T. (2021). "Open Problems in Cooperative AI." *Nature Machine Intelligence*, 3, 299–311.

---

## 3. Multi-Objective RL: Theory of Weight Selection

### 3.1 Linear Scalarization and Its Limits

Hayes et al. (2022) and Roijers et al. (2013) establish the theoretical framework for multi-objective RL:

- **Linear scalarization** (R = Σ w_i × R_i) is the simplest approach but can only find solutions on the **convex hull** of the Pareto front. Concave-region solutions are unreachable.
- **Non-linear scalarization** (Chebyshev, Nash product) can find all Pareto-optimal solutions.
- Our reward function is a **hybrid**: linear in self and welfare terms, non-linear through the Nash product term. This gives us access to a wider region of the Pareto front than pure linear scalarization.

**Reference:** Hayes, C. F., Radulescu, R., Bargiacchi, E., et al. (2022). "A Practical Guide to Multi-Objective Reinforcement Learning and Planning." *Autonomous Agents and Multi-Agent Systems*, 36, 26. arXiv:2103.09568.

**Reference:** Roijers, D. M., Vamplew, P., Whiteson, S., & Daker, R. (2013). "A Survey of Multi-Objective Sequential Decision-Making." *JAIR*, 48, 67–113.

### 3.2 Adaptive and Conditioned Weights

Yang et al. (2019) proposed training a single network **conditioned on preference vectors** (the lambdas) that can generalize across the entire Pareto front. At test time, you choose any weighting without retraining.

This is relevant for our ablation (RQ2): instead of training separate runs per lambda config, we could theoretically train one model conditioned on (lambda_self, lambda_welfare, lambda_fair) and sweep at inference. However, this adds significant implementation complexity and is beyond our current scope.

**Reference:** Yang, R., Sun, X., & Narasimhan, K. (2019). "A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation." *NeurIPS 2019*.

---

## 4. Why Nash Product Deserves the Highest Cooperative Weight

Synthesizing the literature, the Nash product emerges as the strongest single cooperative metric for our setting:

| Property | Self (U_A) | Welfare (U_A + U_B) | Nash (U_A × U_B) |
|----------|-----------|---------------------|-------------------|
| Rewards self-interest | Yes | Partially | Partially |
| Rewards total value creation | No | Yes | Yes (indirectly) |
| Rewards fair splits | No | No | **Yes** |
| Zero when either party gets nothing | No | No | **Yes** |
| Scale invariant | Yes | No | **Yes** |
| Signal in distributive games | Yes | **No** (constant) | Yes |
| Signal in compatible games | Yes | Yes | Yes |
| Axiomatic foundation | — | Bentham (1789) | **Nash (1950)** |
| Envy-freeness guarantee | No | No | **Yes** (Caragiannis 2019) |

The Nash product:
- Provides signal in ALL game types (unlike welfare, which is constant in distributive games)
- Inherently balances efficiency and fairness (Caragiannis et al., 2019)
- Has the strongest axiomatic justification (Nash, 1950)
- Acts as an implicit inequity penalty (like Fehr-Schmidt, but smoother)
- Is zero when negotiations fail (0 × anything = 0), naturally incentivizing agreement

Welfare adds unique value only in integrative/compatible games where the total pie size varies AND the split is already balanced. In our game set, this is a subset of scenarios.

---

## 5. Recommended Lambda Configurations

Based on the literature review, here are the recommended configurations for the next round of runs. All assume normalized components (each in [0, 1]).

### 5.1 Primary Ablation (4 runs)

| Run Name | lambda_self | lambda_welfare | lambda_fair | Rationale |
|----------|-------------|----------------|-------------|-----------|
| **Self-only** | 1.0 | 0.0 | 0.0 | Baseline. Already done (eunj3abz). Equivalent to Lewis et al. (2017), He et al. (2018). |
| **Nash-dominant** | 0.3 | 0.0 | 0.7 | Nash product as primary objective with self-interest as stabilizer. Justified by Caragiannis et al. (2019): Nash welfare alone provides efficiency + fairness. Self at 0.3 keeps the agent from degenerating (Peysakhovich's lower bound). |
| **Balanced (self + nash)** | 0.5 | 0.0 | 0.5 | Clean 50/50 self vs fairness. Tests whether equal self/nash weighting captures "best of both worlds." Aligns with Peysakhovich's finding that ~50% other-regard is the upper bound before exploitability. |
| **Nash-only** | 0.0 | 0.0 | 1.0 | Already done (gvrgl5tx). Pure Nash product optimization. Good for establishing the cooperative ceiling. |

**Why no welfare in the primary ablation?** Nash product already captures both efficiency and fairness (Section 4). Adding welfare dilutes the signal. We test welfare's contribution separately in the secondary ablation.

### 5.2 Secondary Ablation: Does Welfare Add Value? (2 runs)

| Run Name | lambda_self | lambda_welfare | lambda_fair | Rationale |
|----------|-------------|----------------|-------------|-----------|
| **Nash + welfare** | 0.3 | 0.2 | 0.5 | Adds welfare to the Nash-dominant config. Tests whether welfare provides additional signal in integrative/compatible games. |
| **Balanced (all three)** | 0.4 | 0.3 | 0.3 | The "recommended balanced" config. Self slightly dominant for stability, welfare and nash equal. Tests the three-component formula. |

Comparing "Nash-dominant" (0.3/0.0/0.7) vs "Nash + welfare" (0.3/0.2/0.5) directly isolates welfare's contribution.

### 5.3 Why These Specific Numbers?

**lambda_self ≥ 0.3 (when included):**
- Peysakhovich & Lerer (2017) found that self-regard below ~50% (other-regard above ~50%) makes agents exploitable
- Hughes et al. (2018) showed both advantageous and disadvantageous inequity aversion are needed — pure other-regard doesn't work
- Keeps the agent from degenerating into "always concede" strategies

**lambda_fair as dominant cooperative term:**
- Caragiannis et al. (2019): Nash welfare provides efficiency + fairness simultaneously
- Nash (1950): Only social welfare function satisfying all four fairness axioms
- Provides signal in ALL game types including distributive (unlike welfare)
- Our own data: fair-only (eunj3abz) achieved the best agreement rate and cooperative metrics

**lambda_welfare as optional supplement:**
- Only adds unique signal in integrative/compatible games where total pie size varies
- Redundant with Nash product in most scenarios
- Zero variance in ~7% pure distributive games, reduced variance in games with distributive issues
- Include at 0.2–0.3 to test, but don't expect large impact

---

## 6. Alternative Approaches to Consider

### 6.1 Log Nash Product

Instead of `lambda_fair × (U_A × U_B) / max_nash`, consider:

```
lambda_fair × (log(U_A + 1) + log(U_B + 1)) / (log(max_U_A + 1) + log(max_U_B + 1))
```

**Advantage:** The gradient w.r.t. U_A is proportional to 1/U_A — improving a *bad* outcome gets a **strong** signal, while improving an already-good outcome gets a weak signal. This is exactly the fairness-promoting property we want. The raw product's gradient is proportional to U_B, which doesn't have this diminishing-returns property.

**Disadvantage:** More complex, requires handling log(0+1)=0 edge case for failed negotiations. The normalization is also less intuitive.

**Status:** Not implemented. Worth considering for a future run if Nash-dominant underperforms.

### 6.2 Curriculum Learning (Adaptive Lambdas)

Start with high lambda_self (agent first learns to negotiate competently), then gradually increase lambda_fair (agent learns to negotiate cooperatively). This mirrors human negotiation training: first learn to get deals done, then learn to make them fair.

**Literature support:** Yang et al. (2019) showed that conditioning on preference vectors enables smooth interpolation across the Pareto front. A simpler version: linear interpolation of lambdas over training steps.

**Status:** Not implemented. Interesting future direction.

### 6.3 Diverse Opponent Preferences

McKee et al. (2020) showed that diversity of preferences matters more than specific values. Instead of a fixed frozen opponent, sample the opponent's "personality" (cooperative, competitive, balanced) per episode. This would require modifying the opponent's system prompt or using multiple opponent checkpoints.

**Status:** Not implemented. Relevant for RQ4 (robustness).

---

## 7. Summary: Key Literature References

| Paper | Year | Venue | Key Finding for Our Work |
|-------|------|-------|--------------------------|
| Nash, "The Bargaining Problem" | 1950 | Econometrica | Nash product is the unique fair + efficient solution |
| Roijers et al., "Survey of MORL" | 2013 | JAIR | Linear scalarization only finds convex Pareto front |
| Lewis et al., "Deal or No Deal" | 2017 | EMNLP | Foundational negotiation RL; self-interested reward |
| Peysakhovich & Lerer, "Prosocial Learning Agents" | 2017 | arXiv:1709.02865 | ~20–50% other-regard is the sweet spot |
| Cao et al., "Emergent Communication through Negotiation" | 2018 | ICLR | Nash product as evaluation metric (not training reward) |
| He et al., "Decoupling Strategy and Generation" | 2018 | EMNLP | Self-interested reward for negotiation |
| Hughes et al., "Inequity Aversion" | 2018 | NeurIPS | Fehr-Schmidt inequity aversion improves cooperation |
| Caragiannis et al., "Unreasonable Fairness of Max Nash Welfare" | 2019 | ACM EC/TECo | Nash welfare = efficiency + fairness simultaneously |
| Yang et al., "Generalized Algorithm for MORL" | 2019 | NeurIPS | Conditioned networks for Pareto front exploration |
| McKee et al., "Social Diversity and Social Preferences" | 2020 | AAMAS | Diversity of preferences > specific optimal weights |
| Dafoe et al., "Open Problems in Cooperative AI" | 2021 | Nature MI | Research agenda; cooperative AI is under-explored |
| Hayes et al., "Practical Guide to MORL" | 2022 | AAMAS journal | Linear scalarization limits; weight selection methods |

---

## 8. What This Means for the Thesis (RQ2)

**RQ2: How should reward functions balance self-interest vs. collective welfare?**

The literature and our data converge on these answers:

1. **Nash product should be the dominant cooperative term.** It is the only metric that provides signal in ALL game types, has axiomatic justification (Nash, 1950), and simultaneously achieves efficiency and fairness (Caragiannis et al., 2019). Our fair-only run confirms this empirically.

2. **Self-interest should be maintained at ~30–50%.** Below this, agents become exploitable or degenerate (Peysakhovich & Lerer, 2017; Hughes et al., 2018). Above this, cooperative signal is too weak.

3. **Welfare adds limited unique value.** In theory, welfare incentivizes total value creation. In practice, Nash product already captures this in most game types. Welfare only adds unique signal in compatible/integrative games where the total varies but the split is already balanced — a narrow niche.

4. **Equal weighting is unjustified.** There is no theoretical or empirical basis for lambda = (1/3, 1/3, 1/3). Each component has different strengths, weaknesses, and signal properties across game types. Weights should reflect these differences.

5. **Exact weights matter less than having cooperative signal.** McKee et al. (2020) showed that diversity of preferences matters more than specific values. As long as the reward includes some cooperative component, the agent learns cooperative behavior. The ablation's value is in understanding the trade-off, not in finding a single "optimal" config.

6. **Our approach is novel.** Most prior negotiation RL uses pure self-interest (Lewis et al., 2017; He et al., 2018; Cao et al., 2018). Directly incorporating Nash product into the training reward — rather than just evaluating with it post-hoc — is a contribution of this thesis.
