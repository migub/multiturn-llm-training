# From Competition to Cooperation: Teaching LLMs Pareto-Efficient Negotiation with LA-GRPO

**Preliminary Study**

- **Author:** Michael Gubler, Dättnauerstrasse 133, CH-8406 Winterthur, michael.gubler@stud.hslu.ch
- **Supervisor:** Oliver Staubli, Lucerne University of Applied Sciences and Arts, Werftestrasse 4, CH-6002 Lucerne, oliver.staubli@hslu.ch
- **Co-Supervisor:** Peter Niederberger, Tincan AG, Feldpark 17, CH-6300 Zug, p.niederberger@tincan.ch

Lucerne University of Applied Sciences and Arts — Master of Science in Applied Information and Data Science (MScIDS) — Autumn Semester 2025

Winterthur, 11th October 2025

---

## 1 Introduction

Large Language Models (LLMs) have rapidly advanced in their ability to conduct multi-turn dialogues across a variety of domains, ranging from tutoring and healthcare to code generation and strategic interaction. A particularly relevant testbed for studying multi-turn reasoning is negotiation, where agents must balance short-term concessions with long-term gains, maintain coherence across multiple exchanges, and adapt to dynamic counterpart behavior. For instance, GPT-4 and similar models already perform well on various negotiation subtasks (e.g. understanding offers and devising counter-proposals), often matching or surpassing specialized models in objective evaluations (Kwon et al., 2024).

Franceschetti (2025) introduced Local Advantage Group Relative Policy Optimization (LA-GRPO), a novel method that addresses the credit assignment problem in multi-turn dialogues by isolating learning signals at the turn level. LA-GRPO outperformed traditional GRPO in complex bargaining settings, achieving more efficient and effective negotiation strategies without degrading general language capabilities. However, this work primarily focused on zero-sum bargaining scenarios, where one party's gain directly translates into the other's loss.

While zero-sum games are analytically convenient, many real-world negotiations are non-zero-sum. Examples include labor agreements, international treaties, or business partnerships, where cooperation and joint value creation play a critical role. In such contexts, success is not only measured by maximizing individual payoff but also by achieving Pareto-efficient outcomes, where no party can be made better off without making the other worse off. Cooperative negotiation requires LLMs to balance self-interest with fairness, recognize opportunities for mutual gain, and avoid exploitative strategies that may undermine long-term trust or stability.

Most existing research and alignment of LLM-based negotiators has focused on competitive or zero-sum settings. Jiang and Akçakır (2025) noted that in the absence of explicit negotiation instructions, LLMs showed a 50-90% lower success rate than in instructed scenarios. In other words, current LLMs are trained to compete for the largest share, with little attention to fairness or joint outcomes. Human negotiation theory warns of a "negotiator's dilemma," where tactics for claiming value (hard bargaining and withholding information) directly undermine those for creating value (open sharing and building trust). If both parties compete relentlessly, the result is often a dead end or a mediocre compromise, whereas mutual cooperation could have yielded a better Pareto-optimal outcome (Lax and Sebenius, 1992). Notably, a recent AI negotiation competition found that agents using warm and cooperative communication achieved higher joint gains and more agreements, whereas overly dominant agents frequently stalemated (Vaccaro et al., 2025).

Since I believe that AI negotiators will soon enter the real world en masse, it is important to increase value creation for everyone by training LLMs to both compete and cooperate intuitively – without the need of explicit prior instructions.

---

## 2 Content

### 2.1 Thematic Scope

This thesis builds directly on Franceschetti's (2025) work by shifting the focus from competition to cooperation. It investigates how LA-GRPO and related alignment techniques can be adapted to multi-objective reward structures, enabling LLMs to learn negotiation strategies that maximize both individual and collective outcomes. This requires agents to identify creative trade-offs and integrative solutions that improve overall welfare for everyone. The overarching aim is to investigate whether alignment methods originally developed for zero-sum bargaining can be adapted to teach models cooperative behavior - balancing self-interest with collective welfare.

The thematic emphasis on cooperation is not only of academic interest but carries significant real-world relevance. Many of the most pressing negotiation challenges in society are fundamentally non-zero-sum. For example contract negotiations in business benefit from collaborative thinking: long-term partnerships flourish when agreements are seen as fair and value-maximizing for both client and provider, rather than one party squeezing the other.

A central aspect of this exploration is the learning process itself. Franceschetti's thesis showed that applying credit assignment locally at the dialogue-turn level led to more stable and efficient learning in competitive negotiations. Building on that insight, this work examines whether similar turn-level learning dynamics can foster cooperation and therefore allowing the model to recognize the consequences of individual utterances and gradually internalize strategies that promote mutual gain.

The thesis will employ multi-issue negotiation tasks as test environments. Unlike a simple single-issue deal where one side's gain equals the other's loss, multi-issue negotiations allow for trade-offs that benefit both sides. Examples include the Nash Bargaining Game and the Two-Issue Bargaining Problem, which will be explained further in the next chapter.

Finally, cooperation must of course be tempered with robustness to defection. In practice, a negotiator (whether human or AI) should be willing to cooperate but not naïvely so – it must be resilient if the other side attempts to cheat, bluff, or unduly take advantage of goodwill. Therefore, a subtler aspect of the thesis's scope is investigating how LLM agents can maintain a cooperative stance that welcomes mutual trust, but also detect and appropriately respond to uncooperative behavior by others. The aim is to cultivate agents that promote collaboration by default, but can fall back on protective tactics if needed, ensuring they are not easily exploited. This balance between collaboration and caution is critical for real-world deployment: it mirrors the way effective human negotiators build trust and find common ground, while still preparing for worst-case scenarios.

### 2.2 Research Questions

**RQ1: Turn-Level Learning:** Does applying LA-GRPO at the single-turn level improve learning performance in cooperative negotiation games?

This question tests whether applying LA-GRPO at the single-turn level improves learning stability and convergence in cooperative negotiations. Performance will be compared against baseline GRPO by measuring mean negotiation reward, convergence rate, and token efficiency across multiple training runs.

**RQ2: Reward Function Design:** How can reward functions be designed to balance individual payoff maximization with collective welfare and fairness?

This question investigates how different scalar reward compositions affect negotiation outcomes. Various weightings of self-utility, social welfare, and fairness (λ parameters) will be evaluated through ablation studies. Success is indicated by higher joint welfare and Pareto efficiency without sacrificing individual payoff.

**RQ3: Cooperation-Focused Metrics:** Which evaluation metrics best capture cooperative success in negotiation beyond traditional utility scores?

This question examines which evaluation metrics best capture cooperative behavior. In addition to individual utility, outcomes will be assessed using social welfare, Nash product, and fairness indices. Pareto front analysis and equality metrics will be used to identify efficiency–fairness trade-offs.

**RQ4: Robustness to Defection:** Can LLMs trained for cooperation maintain robustness and negotiation effectiveness when facing non-cooperative or adversarial counterparts?

This question assesses whether cooperative agents remain effective when facing non-cooperative or adversarial counterparts. Experiments will pair trained models with selfish baselines and random agents to measure negotiation success rates and payoff degradation under defection.

**RQ5: Impact on LLM Capabilities:** To what extent does promoting cooperation affect the linguistic and reasoning capabilities of the underlying LLM?

This question evaluates whether training for cooperation affects general reasoning and linguistic ability. The fine-tuned models will be tested on standard NLP benchmarks (MMLU-Pro, GLUE, IFEval, EQ-Bench) and compared to the base model to detect possible over-optimization or capability loss.

---

## 3 Content and Methodological Approach

### 3.1 Overview of Approach

The core idea is to teach an LLM to balance individual and collective interests by learning directly from self-generated negotiation dialogues building directly on Franceschetti (2025), the method retains the turn-level credit assignment mechanism - isolating each negotiation turn as an independent optimization unit - but replaces the original competitive payoff with a cooperative scalar reward.

Each training iteration follows the following steps:

1. **Scenario initialization** — Two symmetric instances of the same model (LLaMA 3.1 8B with LoRA adapters) negotiate in natural language. The roles differ only in private payoff tables describing their preferences over issues.

2. **Token generation until random turn h** — The dialogue proceeds token-by-token until a random negotiation turn h is reached. Sampling different turns allows the algorithm to attribute downstream outcome differences to a single utterance and to spread learning evenly across the dialogue horizon.

3. **Forking G local continuations** — At turn h, the model forks G continuations from the identical prefix — generating G alternative offers or responses — and then completes each dialogue to termination (either agreement or no-deal). These local rollouts differ only in what π₀ says at turn h.

4. **Outcome extraction** — Each completed dialogue is parsed by an evaluator model (GPT-4o-mini) that extracts the final agreement and detects whether a valid deal was reached. The evaluator model receives the final agreement as a string and returns a formatted JSON that allows for reward calculations.

5. **Reward computation** — For each continuation, the final agreement is mapped to numeric utilities U_A, U_B using the private payoff tables. The cooperative scalar reward combines individual and collective performance.

6. **Computing local group-relative advantages** — Across the G returns derived from the same context, rewards are normalized to compute local advantages. Each advantage is assigned to all tokens of turn h, tracing performance differences back to that specific utterance.

7. **LA-GRPO update** — The model's LoRA adapters are updated by maximizing the log-probability of the tokens in higher-advantage turns while applying a KL regularization term that constrains the policy toward a frozen reference model. Only π₀ is updated; π_c remains fixed to ensure a stationary environment.

After updating the model's LoRA adapters, a new scenario is initialized. Because new dialogues are generated after every update, the model is constantly learning from its own evolving behavior rather than from a static dataset.

### 3.2 Negotiation Environment and Data Basis

The LAMEN framework provides a rich negotiation environment with modular scenario definitions. Games, issues, agent roles, and negotiation protocol rules are specified through YAML configuration files, which are composed using the Hydra library for dynamic experiment setup (Davidson et al., 2024).

- **Games:** Each game file defines the scenario context and the set of issues and their possible outcomes.
- **Issues:** Every issue is accompanied by private payoff values for each agent. Every payoff value also contains a corresponding label (e.g. value: 10, label: $10). This allows the use and combination of different issues for various types of games.
- **Agent roles:** Agents roles define each of the two LLMs with name, gender, profession, age and personality type. This file provides an internal and external description. The internal description is only accessible to the agent itself, while the external description can also be seen by the opposing agent.
- **Rules:** The rules are based on the Rio Copa game (Bontempo and Iyengar, 2008). It defines prompts such as to never make offers that are not part of the possible values in the payoff table.

Depending on how these payoff values align between the two agents, LAMEN categorizes games into four archetypes:

**Distributive:** All issues are purely competitive (zero-sum) - one agent's gain is the other's loss. For a single-issue distributive game, the agents' interests directly oppose each other (Davidson et al., 2024). A typical example is splitting a fixed resource: if two friends split a pizza, each extra slice for Alice is a slice Bob cannot have.

**Compatible:** All issues are fully aligned – both agents prefer the same outcomes. In a compatible single-issue game, there is no conflict of interest (Davidson et al., 2024). For instance, if both parties benefit from adding extra cheese on a pizza, deciding the cheese amount is a compatible issue.

**Mixture:** Games that include multiple issues with a mix of distributive and compatible (Davidson et al., 2024). For example, a two-issue game might have one issue being distributive (e.g. splitting slices of pizza) and another issue compatible (e.g. both want more cheese). This mixture of issue types means the game has some elements of pure conflict and some of common interest.

**Integrative-Distributive:** Multi-issue games where all issues are in themselves contested (distributive), but the agents attach different importance weights to each issue. This asymmetric preference structure creates opportunities for integrative trade-offs. In an integrative-distributive game, each agent values the issues differently, so they can cooperate by conceding on less-important issues in exchange for gains on more-important issues (Davidson et al., 2024).

This work focuses on mixed-motive games, as they require the agent to navigate competitive dynamics while also finding mutually beneficial compromises.

During the initialization of a scenario, a random game with random issues from a pre-defined set will be chosen. This prevents overfitting to one specific case as it includes different kinds of scenarios with unique payoff tables.

### 3.3 Reward Computation

In Franceschetti's (2025) original formulation, each agent's return is based purely on its individual utility derived from the final negotiated outcome. This setup corresponds to zero-sum or competitive bargaining, where one party's gain is typically the other's loss and learning is driven by self-interest.

In contrast, the present work extends this reward structure to non-zero-sum (mixed-motive) negotiation. Adapting LA-GRPO to this setting requires redefining the scalar reward so that it reflects not only the agent's own payoff but also the collective efficiency and equity of the final outcome.

To formalize this, we introduce three cooperative criteria commonly used in bargaining theory and multi-agent reinforcement learning:

- **Self-utility (U_A):** The agent's own payoff from the agreed outcome. This term preserves an incentive for self-interested performance. This is what was used by Franceschetti (2025).
- **Social welfare (U_A + U_B):** The joint utility of both parties, representing total value creation (Liu et al., 2022). Maximizing this term encourages the agent to find agreements that improve overall efficiency.
- **Fairness / Nash product (U_A × U_B):** A measure introduced by Nash (1950), which reaches its maximum when both parties achieve high and balanced payoffs. The product drops sharply if either side receives little value, implicitly penalizing unequal or exploitative deals and promoting Pareto-efficient and equitable outcomes.

Combining these objectives yields a single scalar cooperative reward used by LA-GRPO to update the trainable policy π₀.

The environment computes both agents' utilities (U_A, U_B) from their payoff tables, but only the learning agent receives the scalarized signal. The reward is calculated by adding up the self-utility, social welfare and nash product:

```
R_coop = λ_self × U_A + λ_welfare × (U_A + U_B) + λ_fair × (U_A × U_B)
```

where `λ_self`, `λ_welfare`, `λ_fair` >= 0 are tunable coefficients balancing self-interest, total welfare and fairness. λ will be adjusted during training to compare different types of reward functions.

When `λ_welfare = λ_fair = 0`, this formulation reduces to Franceschetti's original self-utility reward. Increasing these parameters gradually shifts behavior toward cooperative and Pareto-efficient negotiation.

This cooperative scalarization preserves the single-agent optimization loop required by LA-GRPO while embedding social objectives grounded in bargaining theory and cooperative reinforcement learning. It therefore allows the model to learn cooperation from feedback alone, without explicit prompting or role conditioning.

### 3.4 Calculating Local Advantage

Having defined the cooperative reward R_coop, we can now compute the local advantage signal used by LA-GRPO to assign credit and update the policy.

To isolate the learning signal for a particular dialogue turn, we follow Franceschetti's (2025) Local Advantage Group Relative Policy Optimization procedure. The idea is to sample several continuations from the same dialogue context and measure how each continuation's final reward compares to the group's weighted average reward:

```
Â_h^i = (R(s_{N+1}^i) - mean(R_coop(s_{N+1}^i))) / std(R_coop(s_{N+1}^i)),  i = 1, ..., G
```

This relative difference - rather than the raw return - becomes the advantage estimate for that turn.

### 3.5 Policy Update with LoRA Adapters

After computing the local advantage Â_h^i for each sampled continuation, the trainable parameters of the negotiation model are updated through a gradient-ascent step that increases the likelihood of higher-advantage utterances. The update process follows the same optimization principle introduced by Franceschetti (2025), but uses the local advantage based on R_coop instead of the purely self-interested payoff.

Formally, the objective for the update step is defined as:

```
L = -(1/G) Σ_{i=1}^{G} Σ_{t=1}^{T_i} [ (π_θ(a_{h,t}^i | s_{h,t}^i) / π_old(a_{h,t}^i | s_{h,t}^i)) × Â_h^i + β × D_KL(π_θ(a_{h,t}^i | s_{h,t}^i) ∥ π_ref(a_{h,t}^i | s_{h,t}^i)) ]
```

The first term maximizes the log-probability of tokens belonging to high-advantage turns, therefore reinforcing actions that produced cooperative and Pareto-efficient outcomes. The second term applies KL-regularization to constrain the updated policy toward the frozen reference model π_ref, preventing language drift or reward hacking.

Only a low-rank subset of parameters - the LoRA adapters - is trainable (Hu et al., 2021). This modular fine-tuning strategy allows efficient weight updates on limited hardware (A100 / RTX 4090) while preserving the base model's general linguistic competence. Gradients are computed through backpropagation in PyTorch and accumulated over several dialogue batches before performing the optimizer step.

After each policy update, the LoRA weights are merged with the frozen base model to form a new active policy π_θ^new. The negotiation loop then restarts with fresh self-play dialogues generated from this updated policy, keeping the procedure fully on-policy. Because dialogues and rewards are regenerated after every iteration, the model continuously learns from its own evolving negotiation behaviour.

### 3.6 Training Procedure and Issue Selection

Each training iteration begins by sampling a negotiation configuration g from a predefined set of training games G_train, derived from LAMEN (Davidson et al., 2024) and extended to include both single- and multi-issue setups. The sampling is stratified to maintain a balanced representation of scenarios (rental, loan, merger) and issue types (distributive vs. compatible). This ensures that the model is repeatedly exposed to both competitive and cooperative dynamics.

For multi-issue training, two issues (i₁, i₂) are drawn at random from the selected scenario's issue pool. Each issue has its own payoff table and importance weighting vector (ω_{A,1}, ω_{A,2}) sampled from {(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)}. This variation forces the agent to identify which issue carries higher value in each game and to negotiate trade-offs accordingly. One issue may be distributive, while the other is compatible, resulting in a mixed-motive negotiation that combines competitive and cooperative objectives within a single dialogue.

At the start of each episode, the initial prompt s₀ is constructed by inserting the scenario description, role information, and payoff tables of the sampled issues. The model then conducts a full negotiation consisting of five rounds per agent (N = 5), alternating roles across episodes to avoid first-mover bias. During LA-GRPO training, the turn h at which the dialogue diverges into multiple continuations is drawn from a geometric distribution h ~ Geom(0.3), prioritizing early turns where strategic anchors typically emerge (Franceschetti, 2025).

Each training batch contains several dialogue rollouts sampled on-policy from the current model π_θ. After reward extraction and advantage estimation, gradients are accumulated over a fixed number of batches and the LoRA adapters are updated as described in Section 3.5.

### 3.7 Evaluation of the Final Model

The trained models are evaluated on both seen and unseen negotiation games to measure cooperative performance, robustness, and general language capability. All evaluation follows the LAMEN framework (Davidson et al., 2024), ensuring compatibility with Franceschetti (2025).

**Negotiation Outcome Metrics.** For each dialogue, the final agreements on all negotiated issues are parsed by the outcome extractor, and agent-specific rewards are computed as in Eq. (1). From these values, additional aggregate metrics are derived.

The joint welfare captures total efficiency, i.e. how much total value the negotiation produced for both agents:

```
W = R_A + R_B
```

The Nash product measures fairness and balance between the agents, rewarding outcomes where both achieve high payoffs simultaneously:

```
N = R_A × R_B
```

**Example evaluation table (simplified, good scenario):**

| Model | R_A | R_B | W = R_A + R_B | N = R_A × R_B |
|---|---|---|---|---|
| Base (LLaMA 3.1 8B) | 45.3 | 40.7 | 86.0 | 1,844 |
| GRPO (zero-sum) | 52.4 | 37.9 | 90.3 | 1,986 |
| LA-GRPO (zero-sum) | 54.6 | 36.2 | 90.8 | 1,977 |
| LA-GRPO (ours) | 56.8 | 49.1 | 105.9 | 2,789 |

Mean negotiation outcomes across models on the mixed-motive multi-issue setup. Each value represents the average over 200 negotiation episodes.

**Robustness and Comparative Evaluation.** Each model - the base LLaMA 3.1 8B, GRPO baseline, and cooperative LA-GRPO variant - is evaluated against fixed, selfish, and random opponents to assess adaptability under mixed-motive conditions such as iterated prisoner's dilemma, following Duan et al. (2024). For each configuration, 100–200 episodes are sampled under identical random seeds. Mean reward, agreement rate, and tokens per reward point are reported with 95% bootstrap confidence intervals.

**Language and Generalization Tests.** To verify that alignment on negotiation tasks does not harm general capabilities, the final models are re-evaluated on MMLU-Pro (Wang et al., 2024), GLUE (Wang et al., 2019), IFEval (Zhou et al., 2023), EQ-Bench (Paech, 2024), and LM-Pragmatics (Hu et al., 2023).

**Success Criteria.** A model is considered successful if it achieves higher joint welfare and Nash product than GRPO, maintains comparable benchmark accuracy to the base model, and produces coherent, cooperative dialogues without degeneration.

### 3.8 Project Methodology

The project follows a hybrid iterative methodology that combines elements of Scrum with a structured research plan. Instead of a static waterfall sequence, experiments are conducted in short two-week iterative cycles: each cycle involves implementing small changes (e.g., reward parameters), running training, evaluating outcomes, and refining the setup. Sprint progress and tasks are managed in Asana to ensure transparent tracking of goals and milestones.

This reflects the exploratory nature of reinforcement-learning research, where incremental adjustments and frequent validation are essential.

---

## 4 Technology, Software and Applications

All experiments are executed on Google Colab using A100 or RTX 4090-class GPUs, with the provided custom LA-GRPO implementation forming the training backbone. If there are problems with Google Colab such as memory constraints, another service like AWS or (if available) a cluster from a university will be used. The following setup is in large parts done similar to the work of Franceschetti (2025) to ensure consistency.

### 4.1 Model Architecture

The negotiation agents are based on LLaMA 3.1 8B (Grattafiori and et al., 2024), selected for its balance between performance and resource requirements. It successfully executes full multi-turn dialogues with meaningful reward signals while fitting on a single high-end GPU. To adapt the model efficiently, Low-Rank Adaptation (LoRA) modules (Hu et al., 2021) are applied with rank 8, α = 16, and dropout = 0.1. LoRA reduces the number of trainable parameters by several orders of magnitude while preserving model quality.

### 4.2 Training Framework

The learning process builds directly on Local Advantage Group Relative Policy Optimization (LA-GRPO), Franceschetti's (2025) reinforcement learning method designed for multi-turn alignment. LA-GRPO improves credit assignment by sampling multiple continuations from the same dialogue state and computing turn-level local advantages based on outcome differences. This enables the model to link rewards to specific utterances rather than entire dialogues, resulting in more stable and data-efficient learning. The implementation provided by Franceschetti (2025) is reused without modification to its optimization core, ensuring methodological consistency. Updates are applied to LoRA adapters via PyTorch and the Transformers library, maintaining compatibility with standard reinforcement-learning interfaces. The training loop is on-policy: new dialogues are generated each iteration, rewards are computed, and the policy is updated immediately. This yields fast adaptation and avoids stale data issues common in offline fine-tuning.

### 4.3 Dialogue Generation and Evaluation

For efficient large-scale sampling, inference is handled through vLLM (Kwon et al., 2023), a high-throughput engine optimized for GPU memory management using paged attention. This setup supports concurrent generation of multiple dialogue continuations on a single Colab GPU, significantly reducing training latency. Dialogue evaluation and outcome extraction are conducted by GPT-4o-mini (OpenAI, 2024), which automatically parses negotiation transcripts and identifies final agreements. Franceschetti (2025) verified that this evaluator achieved 98% accuracy across 500 samples, making it reliable for reward computation. Using GPT-4o-mini via API also reduces local memory requirements and ensures consistency across runs. Together, vLLM and GPT-4o-mini enable an automated loop of dialogue generation, structured evaluation, and immediate policy updates.

### 4.4 Negotiation Environment

The negotiation tasks are defined using the LAMEN framework (Davidson et al., 2024), which provides standardized, YAML-based multi-issue bargaining games. Each scenario specifies roles, issues, and payoff tables that determine agents' private utilities. LAMEN's modular structure allows easy configuration of integrative and mixed-motive games, essential for testing cooperation. Scenario definitions are loaded dynamically using the Hydra configuration system, which supports hierarchical experiment management and parameter overrides. The turn-taking protocol and evaluation prompts follow Franceschetti's original implementation and LAMEN's reference design, ensuring full comparability.

### 4.5 Experiment Configuration and Infrastructure

All components are orchestrated through Hydra configuration files, allowing reproducible control over parameters such as the number of rollouts (G), learning rate, and KL-regularization coefficient. PyTorch Lightning modules are used for structured training loops, simplifying checkpointing and metric logging. Model weights, configuration files, and generated dialogues are version-controlled using GitHub, ensuring reproducibility and transparent documentation of all experiments. Intermediate checkpoints are periodically saved to Google Drive to handle Colab's session limits. Lightweight experiment tracking is achieved via TensorBoard, which records loss, local advantages, KL-divergence, and negotiation success rate. This enables real-time monitoring of model behavior and early detection of instability.

---

## 5 Commented Disposition

Here is a list of chapters that will be included in the master's thesis.

### Chapter 1: Introduction (≈ 4 pages)

Introduces the motivation and relevance of teaching LLMs cooperative negotiation behavior. Establishes the gap between zero-sum bargaining and real-world cooperative scenarios. Defines objectives, hypotheses, and research questions. Ends with an overview of the thesis structure.

- 1.1 Background and Motivation — Outlines the rise of LLM-based negotiation systems and their current limitations.
- 1.2 Research Gap and Objectives — Positions this work as an extension from competitive to cooperative negotiation.
- 1.3 Research Questions — Lists RQ1–RQ5 focusing on learning dynamics, reward design, cooperation metrics, robustness, and generalization.
- 1.4 Thesis Structure — Summarizes upcoming chapters and their roles.

### Chapter 2: Theoretical Background and Related Work (≈ 7 pages)

Provides the conceptual foundation of the thesis. Connects negotiation theory, cooperative game theory, and reinforcement learning alignment. Identifies the limitations of current methods in promoting cooperation.

- 2.1 Negotiations as Multi-Agent Decision Processes — Summarizes Franceschetti's MDP formulation, highlighting how non-zero-sum extensions differ.
- 2.2 Cooperative and Integrative Negotiation Theory — Introduces Pareto efficiency, Nash product, fairness, and integrative bargaining principles.
- 2.3 Alignment Methods for Multi-Turn Dialogue — Reviews DPO, GRPO, REFUEL, and LA-GRPO; explains credit assignment issues.
- 2.4 Cooperative Reinforcement Learning and Social Objectives — Discusses social welfare maximization and fairness-oriented reward shaping.
- 2.5 Summary and Research Gap — Concludes with how adapting LA-GRPO to cooperative objectives addresses the open problem.

### Chapter 3: Methodology: Adapting LA-GRPO for Cooperative Negotiation (≈ 9 pages)

Describes the main technical contribution. Details how the LA-GRPO framework is modified for cooperative reward signals, with clear conceptual and implementation mapping to Franceschetti's setup.

- 3.1 Overview of LA-GRPO — Revisits on-policy design and local advantage estimation.
- 3.2 Cooperative Reward Design — Defines the scalarized objective R_coop combining self-utility, joint welfare, and fairness. Discusses theoretical rationale and parameter weighting.
- 3.3 Local Advantage Computation — Explains how advantages are normalized within continuation groups to assign turn-level credit.
- 3.4 Policy Update with LoRA Adapters — Describes gradient step, KL constraint, and stability aspects; cites Franceschetti's training codebase.
- 3.5 Expected Effects and Hypotheses — States expected outcomes: increased Pareto efficiency and fairness without loss of fluency.

### Chapter 4: Experimental Setup and Data Basis (≈ 7 pages)

Describes how experiments are designed to test the cooperative LA-GRPO framework. Covers the LAMEN negotiation environment, issue sampling, evaluation pipeline, and ethical considerations.

- 4.1 Negotiation Environment — Overview of the LAMEN-based setup and its configuration.
- 4.2 Game and Issue Design — Explains mixed-motive games and random scenario selection to prevent overfitting.
- 4.3 Agent Roles and Initialization — Defines the two symmetric LLM agents and their private payoff structures.
- 4.4 Outcome Extraction and Evaluation — Describes GPT-4o-mini as evaluator and reward JSON formatting.
- 4.5 Technical Setup — Specifies hyperparameters, temperature, G samples, and hardware environment.
- 4.6 Ethical and Practical Considerations — Notes transparency, safety, and social alignment relevance.

### Chapter 5: Results and Discussion (≈ 9 pages)

Presents results, compares baselines, and interprets the cooperative model's performance. Analyzes efficiency, fairness, robustness, and language integrity.

- 5.1 Quantitative Results — Reports mean utilities, joint welfare, and convergence trends.
- 5.2 Pareto and Fairness Metrics — Plots Pareto frontiers and Nash welfare improvements.
- 5.3 Robustness to Defection — Tests against adversarial or selfish counterparts and interprets adaptability.
- 5.4 Language Quality and Benchmark Generalization — Evaluates MMLU-Pro, GLUE, and IFEval; checks for degeneracy.
- 5.5 Comparative Discussion — Synthesizes results and connects them to theoretical expectations.

### Chapter 6: Conclusion and Future Work (≈ 4 pages)

Summarizes findings, theoretical contributions, and implications for AI negotiation. Reflects on the success of cooperative adaptation of LA-GRPO and identifies next research directions.

- 6.1 Summary of Findings — Highlights main contributions and empirical outcomes.
- 6.2 Limitations — Discusses model scope, computational limits, and evaluation constraints.
- 6.3 Future Directions — Suggests multi-agent self-play, multi-round reward shaping, and real-world testing.
- 6.4 Broader Implications — Reflects on societal impact and ethical alignment of cooperative AI negotiators.

Total: roughly 40 pages (excluding Appendix).

---

## 6 Project Risks

### R1 – Compute Limits

Google Colab will serve as the primary compute platform, but its session limits, memory caps, and GPU quotas may constrain training. Should these constraints arise, fallback options such as AWS or an institutional cluster are planned. Even with high-end GPUs, multi-turn reinforcement learning is computationally intensive: each parameter change (e.g. reward weight λ or learning rate) requires retraining on hundreds of dialogues, risking long iteration cycles and potential schedule delays.

### R2 – Integration Complexity

The experimental pipeline combines multiple frameworks: vLLM for efficient inference (Kwon et al., 2023), Hydra for configuration management, PyTorch Lightning for structured training, LoRA for parameter-efficient fine-tuning (Hu et al., 2021), and the LAMEN negotiation environment (Davidson et al., 2024). Any of these could become a bottleneck. For instance, vLLM's paged-attention kernel may behave differently across GPU types, and Hydra configuration mismatches could silently alter reward weights.

### R3 – Reward Design Instability

Extending LA-GRPO from zero-sum to cooperative negotiation requires careful reward design. The scalar reward `R_coop = λ_self × U_A + λ_welfare × (U_A + U_B) + λ_fair × (U_A × U_B)` balances self-interest and fairness, but if λ-weights are poorly tuned, agents may converge to trivial or unstable equilibria. Mis-specified rewards can drive "reward hacking," where models exploit loopholes rather than truly cooperating. Models trained under sparse, end-of-dialogue rewards can learn to maximize numeric scores while drifting from intended behavior - a pattern noted in multi-turn alignment studies (Jang et al., 2024). In practice, this could appear as agents that always agree on mid-range offers or echo each other's phrasing to guarantee high "fairness" without genuine strategic reasoning. Careful inspection of dialogue transcripts and inclusion of qualitative analyses will therefore be essential.

### R4 – Evaluation Reliability

Evaluation risks also merit attention. Negotiation outcomes are parsed automatically by GPT-4o-mini, which showed roughly 98% extraction accuracy in prior tests (Franceschetti, 2025). Minor parsing errors could nonetheless distort computed rewards or fairness metrics. Moreover, standard quantitative indicators such as joint welfare or the Nash product capture efficiency and equity but not conversational warmth or trust. In a large-scale negotiation competition, agents displaying more cooperative and empathetic communication achieved significantly higher joint gains than purely outcome-focused agents (Vaccaro et al., 2025). Thus, an over-reliance on scalar metrics might overstate success if linguistic cooperation deteriorates.

### R5 – Language Degeneration

Franceschetti (2025) observed that LA-GRPO preserved general language capabilities after fine-tuning, yet reinforcement learning methods like PPO or GRPO are known to induce over-optimization or "language drift" when hyperparameters are mis-set (Jang et al., 2024; Ouyang et al., 2022). Degeneration may manifest as repetitive phrasing, typographical artifacts, or overly literal negotiation styles. Monitoring linguistic quality through benchmarks such as MMLU-Pro, EQ-Bench, and LM-Pragmatics will help detect such issues early. Nonetheless, there remains a residual risk of subtle misalignment, where models adopt manipulative or overly submissive negotiation tactics.

### R6 – External Dependencies

The project also depends on several external services and APIs. Colab's availability and GPU quotas may fluctuate; the OpenAI API for GPT-4o-mini could experience downtime or pricing changes. File synchronization across GitHub and Google Drive adds reproducibility but also the risk of version conflicts or data loss. Each external dependency introduces potential disruptions outside of control.

### R7 – Timeline Delays

The project spans seven months - from October 2025 to May 2026 - with multiple interdependent phases. If reproducing the baseline experiments of Franceschetti (2025) or implementing new LAMEN scenarios proves slower than expected, subsequent phases (reward design, training, evaluation) may compress. Since each LA-GRPO training run can take several days, even minor configuration errors could consume significant time. Mitigation includes starting with small-scale prototypes, incremental hyperparameter tuning, and early validation of environment scripts.

---

## 7 Work and Research Plan

The planned research runs from October 2025 to the final submission deadline on 29 May 2026, comprising approximately seven months.

### Phase 1: Preparation and Literature Review (Oct–Nov 2025)

Goal: Establish theoretical foundations and finalize technical setup.

- Conduct an in-depth literature review on cooperative game theory, social welfare optimization, and multi-agent reinforcement learning.
- Reproduce Franceschetti's LA-GRPO experiments in zero-sum settings to ensure reproducibility.
- Set up the computational environment (Google Colab, GitHub, LAMEN framework, Hydra configuration).
- Finalize experimental design and reward formulations (λ parameters for self-utility, welfare, and fairness).

**Deliverables:** Confirmed experimental pipeline, baseline LA-GRPO model, and draft of the theoretical chapter.

### Phase 2: Environment Design and Reward Function Development (Dec 2025–Jan 2026)

Extend the negotiation environment for cooperative and mixed-motive games.

- Implement new YAML-based game configurations in LAMEN (multi-issue, mixed-motive).
- Define and validate payoff tables for both cooperative and competitive issue combinations.
- Implement cooperative reward function `R_coop = λ_self × U_A + λ_welfare × (U_A + U_B) + λ_fair × (U_A × U_B)`
- Conduct ablation tests varying λ parameters to isolate effects of fairness and welfare weighting.

**Deliverables:** Functional cooperative game environment and reward computation module.

### Phase 3: Training and Experimentation (Feb–Mar 2026)

Train negotiation agents using adapted LA-GRPO and baselines.

- Execute self-play training on LLaMA 3.1 8B models with LoRA adapters.
- Compare standard GRPO, LA-GRPO, and cooperative LA-GRPO variants.
- Track performance metrics (mean reward, joint welfare, Nash product, convergence stability).
- Monitor training curves and model stability via TensorBoard.

**Deliverables:** Trained models, intermediate evaluation results, and visualized learning trends.

### Phase 4: Evaluation and Analysis (Apr 2026)

Assess negotiation performance, fairness, and language preservation.

- Evaluate agents on unseen multi-issue games (held-out test set).
- Analyze robustness against non-cooperative opponents.
- Conduct qualitative dialogue assessments (coherence, strategic reasoning, fairness).
- Run benchmark tests (MMLU-Pro, GLUE, IFEval, EQ-Bench, LM-Pragmatics) to detect over-optimization.

**Deliverables:** Comparative performance tables, Pareto frontiers, fairness plots, and benchmark results.

### Phase 5: Writing and Finalization (May 2026)

Complete the written thesis and finalize documentation.

- Integrate empirical findings into the Results and Discussion and Conclusion chapters.
- Validate all citations, figures, and appendix content.
- Conduct proofreading and supervisor review.

**Deliverables:** Final thesis document and submission by 29 May 2026.

---

## 8 Annotated and Structured Literature List

### 8.4.1 Alignment and Reinforcement Learning Methods

- **Franceschetti, L. (2025).** *Using the Advantage: Teaching LLMs to Negotiate in Multi-Turn Dialogues with Local Advantage GRPO.* — Introduces LA-GRPO, a RL alignment method that improves credit assignment in multi-turn dialogues and achieves higher stability than GRPO in zero-sum negotiations.

- **Rafailov, R. et al. (2023).** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* — Proposes DPO, a preference-based post-training method that optimizes LLMs from human comparison data without explicit reward models; a baseline for later multi-turn approaches.

- **Shao, Z. et al. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* — Introduces Group Relative Policy Optimization (GRPO), an online RL method comparing grouped trajectories instead of using a value function; the precursor to LA-GRPO.

- **Gao, Z. et al. (2024).** *Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF.* — Presents REFUEL, a turn-level RL approach that isolates individual dialogue turns to propagate final rewards, partially solving the credit-assignment problem.

### 8.4.2 Negotiation and Cooperation in LLMs

- **Davidson, T. R. et al. (2024).** *Evaluating Language Model Agency through Negotiations.* — Introduces the LAMEN benchmark — a standardized negotiation environment for multi-issue bargaining with human-like dialogue structures and clear outcome metrics.

- **Kwon, D. et al. (2024).** *Are LLMs Effective Negotiators? Systematic Evaluation of the Multi-Faceted Capabilities of LLMs in Negotiation Dialogues.* — Provides a detailed empirical analysis of GPT-4 and other models in multi-issue negotiations, identifying current strengths and weaknesses in coherence, fairness, and strategy.

- **Jiang, Y. & Akçakır, G. (2025).** *Explicit Cooperation Shapes Human-Like Multi-Agent LLM Negotiation.* — Empirically shows that explicit cooperation cues substantially improve negotiation success and human-likeness in LLM interactions.

- **Duan, J. et al. (2024).** *GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations.* — Evaluates LLMs' strategic reasoning abilities across diverse game-theoretic environments, highlighting limitations in long-term planning and strategic adaptation.

- **Vaccaro, M. et al. (2025).** *Advancing AI Negotiations: New Theory and Evidence from a Large-Scale Autonomous Negotiations Competition.* — Reports large-scale evidence that agents using cooperative and empathetic language achieve higher joint gains than purely competitive agents.

### 8.4.3 Game Theory and Bargaining Foundations

- **Nash, J. F. (1950).** *The Bargaining Problem.* — Introduces the Nash bargaining solution, defining fair and Pareto-efficient outcomes as the product of agents' utilities — foundational for cooperative negotiation and fairness metrics.

- **Lax, D. A. & Sebenius, J. K. (1992).** *The Negotiator's Dilemma: Creating and Claiming Value.* — Describes the tension between competitive value claiming and cooperative value creation, explaining why balanced strategies outperform purely self-interested ones.

- **Liu, Z. et al. (2022).** *Welfare Maximization in Competitive Equilibrium: Reinforcement Learning for Markov Exchange Economy.* — Applies RL techniques to model fairness and efficiency trade-offs in multi-agent economic systems, connecting game-theoretic welfare optimization with machine learning.

---

## References

- Bontempo, R. and Iyengar, S. (2008). Rio Copa: A Negotiation Simulation. Columbia Caseworks.
- Davidson, T. R., Veselovsky, V., Josifoski, M., Peyrard, M., Bosselut, A., Kosinski, M., and West, R. (2024). Evaluating language model agency through negotiations.
- Duan, J., Zhang, R., Diffenderfer, J., Kailkhura, B., Sun, L., Stengel-Eskin, E., Bansal, M., Chen, T., and Xu, K. (2024). GTBench: Uncovering the strategic reasoning limitations of LLMs via game-theoretic evaluations.
- Franceschetti, L. (2025). Using the advantage: Teaching LLMs to negotiate in multi-turn dialogues with local advantage GRPO. Master's thesis, ETH Zurich.
- Grattafiori, A. and et al., A. D. (2024). The LLaMA 3 herd of models.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). LoRA: Low-rank adaptation of large language models.
- Hu, J., Floyd, S., Jouravlev, O., Fedorenko, E., and Gibson, E. (2023). A fine-grained comparison of pragmatic language understanding in humans and language models. ACL 2023.
- Jang, Y., Kim, G.-H., Kim, B., Kim, Y. J., Lee, H., and Lee, M. (2024). Degeneration-free policy optimization: RL fine-tuning for language models without degeneration. ICML 2024.
- Jiang, Y. and Akçakır, G. (2025). Explicit cooperation shapes human-like multi-agent LLM negotiation. ICWSM Workshops 2025.
- Kwon, D., Weiss, E., Kulshrestha, T., Chawla, K., Lucas, G., and Gratch, J. (2024). Are LLMs effective negotiators? EMNLP 2024.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention.
- Lax, D. and Sebenius, J. (1992). The manager as negotiator: The negotiator's dilemma.
- Liu, Z., Lu, M., Wang, Z., Jordan, M., and Yang, Z. (2022). Welfare maximization in competitive equilibrium: Reinforcement learning for Markov exchange economy. ICML 2022.
- Nash, J. F. (1950). The bargaining problem. Classics in Game Theory.
- OpenAI (2024). GPT-4o mini: advancing cost-efficient intelligence.
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback.
- Paech, S. J. (2024). EQ-Bench: An emotional intelligence benchmark for large language models.
- Rafailov, R. et al. (2023). Direct Preference Optimization.
- Shao, Z. et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.
- Vaccaro, M., Caosun, M., Ju, H., Aral, S., and Curhan, J. R. (2025). Advancing AI negotiations.
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. (2019). GLUE: A multi-task benchmark.
- Wang, Y. et al. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark.
- Zhou, J. et al. (2023). Instruction-following evaluation for large language models.
