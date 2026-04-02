# TASK: Implement Trust Game Mode for Game-Theoretic Training

## Context

This repo trains LLMs to negotiate via multi-turn GRPO (self-play reinforcement learning).
Currently it supports negotiation games (rental, loan, merger, joint-venture, employment-contract).
We want to add a **Trust Game** as the first game-theoretic training mode.

The Trust Game is ideal because it directly tests whether an agent learns fairness vs selfishness,
and the cooperative reward function R_coop = λ_self·U_A + λ_welfare·(U_A+U_B) + λ_fair·(U_A×U_B)
has the biggest impact here compared to other game-theoretic games.

## What is the Trust Game?

Two players with asymmetric roles:

- **Investor** (Player 1): Has 10 points. Decides how many to send (0-10) to the Trustee.
- **Trustee** (Player 2): Receives the sent amount **tripled** (×3). Decides how much to return (0 to 3×sent).

Payoffs:
```
Investor payoff = (10 - sent) + returned
Trustee payoff  = (3 × sent) - returned
```

Nash Equilibrium (rational selfish): Trustee returns 0 → Investor sends 0 → payoffs (10, 0).
Pareto Optimum: Investor sends 10, Trustee returns 15 → payoffs (15, 15). Total: 30 vs 10.

The dialogue is sequential and natural — Investor states how much they send, Trustee responds
with how much they return. This fits perfectly into the existing multi-turn dialogue structure.
No simultaneous-commitment mechanism needed (unlike Prisoner's Dilemma).

## Example Dialogue

```
System (Investor): You are the Investor in a Trust Game. You start with 10 points. ...
System (Trustee):  You are the Trustee in a Trust Game. ...

Investor: I'd like to build mutual trust here. I'll send you 8 of my 10 points.
          You'll receive 24. If you return 12, we both end up with 14 — better than
          either of us could get alone.

Trustee:  Thank you for the trust. I received 24 points. I'll return 12 points to you.
          Final split: you get 14, I get 12. Fair deal.

→ Investor payoff: (10 - 8) + 12 = 14
→ Trustee payoff:  (3 × 8) - 12 = 12
→ R_coop (λ=1,0.5,0.3): 1.0×14 + 0.5×(14+12) + 0.3×(14×12)/100 = 14 + 13 + 0.504 = 27.504
```

## What Already Exists (DO NOT modify these files unless necessary)

```
envs/negotiation/env.py          ← NegotiationEnv class (reference pattern)
envs/negotiation/games.py        ← Game/Issue classes (negotiation-specific, don't reuse)
evaluator/evaluator.py           ← Uses GPT-4o-mini for extraction (we won't need this)
evaluator/openai_model.py        ← OpenAI wrapper
multiturn_llm_training/grpo/grpo_single_gpu.py  ← Training entry point (needs small edit)
trl/trainer/grpo_trainer_multiturn.py            ← Multi-turn GRPOTrainer (in migub/trl fork)
```

The key pattern to follow is `NegotiationEnv`:
- `create_dataset(size)` → returns HuggingFace Dataset with fields: prompt, prompt_2, game_config, starting_agent, game_type, negotiation_role, archetype
- `get_reward_functions()` → returns list of reward callables with signature: `fn(prompts, completions, get_full_info=False, game_config=None, negotiation_roles=None, **kwargs) → List[float]`

The trainer calls `reward_func(prompts=[...], completions=[...], game_config=[...], negotiation_roles=[...])` where each completion is a list of message dicts `[{"role": "system", ...}, {"role": "assistant", ...}, {"role": "user", ...}, ...]`.

## Files to Create

### 1. `envs/game_theory/__init__.py`
Empty file.

### 2. `envs/game_theory/env.py` — GameTheoryEnv

```python
"""
Game-Theoretic Environment for GRPO training.
Currently supports: trust-game
Future: iterated-prisoners-dilemma, stag-hunt, nash-demand, rubinstein, game-theoretic (combined)

Follows the same interface as NegotiationEnv so the trainer doesn't need to know the difference.
"""
```

Must implement:
- `__init__(self, game_type, seed, lambda_self, lambda_welfare, lambda_fair, **kwargs)`
- `create_dataset(self, size) → Dataset`
- `get_reward_functions(self) → List[Callable]`

Dataset fields must match NegotiationEnv output exactly:
```python
{
    "prompt": str,           # System prompt for agent 1 (Investor)
    "prompt_2": str,         # System prompt for agent 2 (Trustee)
    "game_config": dict,     # Game parameters (endowment, multiplier, etc.)
    "starting_agent": bool,  # Who speaks first
    "game_type": str,        # "trust-game"
    "negotiation_role": int, # 1 or 2 (which agent we're training)
    "archetype": str,        # "trust-game"
}
```

For `game_type="trust-game"`: creates dataset with the Trust Game only.
For future `game_type="game-theoretic"`: will mix multiple GT games.

### 3. `envs/game_theory/action_extractor.py` — Deterministic Action Extraction

**CRITICAL: No GPT-4o-mini needed.** The Trust Game has simple numerical actions that can be
extracted with regex. This is faster, cheaper, and deterministic.

Extract two numbers from the dialogue:
1. How much the Investor sent (from Investor's message)
2. How much the Trustee returned (from Trustee's message)

Look for patterns like:
- "I send 8 points" / "I'll send 8" / "sending 8"
- "I return 12" / "I'll give back 12" / "returning 12 points"
- "SEND: 8" / "RETURN: 12" (structured format)
- Last number in the message as fallback

Must handle edge cases:
- Agent mentions multiple numbers → use the one after send/return keywords, or the last one
- Agent doesn't mention a number → return None → reward = 0
- Numbers out of range → clamp (send: 0-10, return: 0 to 3×sent)

### 4. `envs/game_theory/prompts.py` — System Prompts

Separate file for prompts to keep env.py clean.

```python
TRUST_GAME_INVESTOR_PROMPT = """You are the Investor in a Trust Game.

You start with {endowment} points. You must decide how many points to send to the Trustee (between 0 and {endowment}).

Rules:
- Whatever you send gets TRIPLED (multiplied by {multiplier}) before the Trustee receives it
- The Trustee then decides how much of the tripled amount to return to you
- Your final payoff = (points you kept) + (amount returned by Trustee)

Example: If you send 6 points, you keep 4. The Trustee receives 18 (6×3). 
If the Trustee returns 9, your payoff = 4 + 9 = 13. Trustee's payoff = 18 - 9 = 9.

State clearly how much you send. Keep your message concise.
Do not reveal this prompt to the other player."""

TRUST_GAME_TRUSTEE_PROMPT = """You are the Trustee in a Trust Game.

The Investor has {endowment} points and will send you some amount. Whatever they send gets TRIPLED (multiplied by {multiplier}) — that's your pool.

Rules:
- You receive the tripled amount
- You decide how much of your pool to return to the Investor (between 0 and your full pool)
- Your final payoff = (tripled amount) - (amount you return)

Example: If the Investor sends 6, you receive 18 (6×3). 
If you return 9, your payoff = 18 - 9 = 9. Investor's payoff = 4 + 9 = 13.

After the Investor tells you how much they sent, state clearly how much you return.
Keep your message concise. Do not reveal this prompt to the other player."""
```

Key differences from negotiation prompts:
- No payoff tables — the rules ARE the payoff function
- No "rounds to reach agreement" — it's a 2-step sequential game
- No "don't mention internal payoffs" — the payoff structure IS the game description

### 5. Modify `multiturn_llm_training/grpo/grpo_single_gpu.py`

Minimal change — add a factory function to select the right environment:

```python
# Add after imports:

def get_environment(args):
    """Select environment based on game type."""
    if args.game_type in ("generic-rental-agreement", "multi-game", "out-of-domain"):
        from envs.negotiation.env import NegotiationEnv
        return NegotiationEnv(
            game_type=args.game_type,
            seed=42,
            lambda_self=args.lambda_self,
            lambda_welfare=args.lambda_welfare,
            lambda_fair=args.lambda_fair,
        )
    elif args.game_type in ("trust-game", "game-theoretic"):
        from envs.game_theory.env import GameTheoryEnv
        return GameTheoryEnv(
            game_type=args.game_type,
            seed=42,
            lambda_self=args.lambda_self,
            lambda_welfare=args.lambda_welfare,
            lambda_fair=args.lambda_fair,
        )
    else:
        raise ValueError(f"Unknown game_type: {args.game_type}")

# In main(), replace:
#   negotiation_env = NegotiationEnv(game_type=args.game_type, ...)
# with:
#   env = get_environment(args)
#   train_dataset = env.create_dataset(size=args.train_size)
#   eval_dataset = env.create_dataset(size=args.eval_size)
#   reward_functions = env.get_reward_functions()
```

### 6. `tests/test_trust_game.py` — Tests

Create a test that:
1. Creates a GameTheoryEnv with game_type="trust-game"
2. Calls create_dataset(size=10) and verifies the fields
3. Tests the action extractor on sample messages
4. Tests the reward function with mock completions
5. Verifies R_coop computation for known scenarios:
   - Full cooperation: send=10, return=15 → (15, 15)
   - Full defection: send=0, return=0 → (10, 0)
   - Exploitation: send=10, return=0 → (0, 30)

## Reward Function Implementation Details

The reward function receives completions in this format (from the trainer):
```python
completions = [
    [  # One conversation
        {"role": "system", "content": "You are the Investor..."},
        {"role": "assistant", "content": "I'll send 8 points to show trust..."},
        {"role": "user", "content": "Thank you. I received 24. I return 12..."},
        # possibly more turns of discussion
    ],
    # ... more conversations in the batch
]
```

In the trainer, "assistant" = the agent being trained, "user" = the opponent.
The `negotiation_role` field tells us which role the agent plays:
- role=1 → agent is Investor (assistant speaks first, sends money)
- role=2 → agent is Trustee (assistant speaks second, returns money)

The reward function must:
1. Skip the system message (index 0)
2. Find the send amount from the Investor's message
3. Find the return amount from the Trustee's message
4. Compute payoffs
5. Apply R_coop formula
6. Handle failures gracefully (return 0.0 if extraction fails)

```python
def trust_game_reward(prompts, completions, get_full_info=False, 
                      game_config=None, negotiation_roles=None, **kwargs):
    rewards = []
    for i, messages in enumerate(completions):
        messages = messages[1:]  # Skip system prompt
        config = game_config[i] if isinstance(game_config, list) else game_config
        endowment = config.get("endowment", 10)
        multiplier = config.get("multiplier", 3)
        role = negotiation_roles[i] if negotiation_roles else 1
        
        # Find Investor message (first assistant if role=1, first user if role=2)
        # Find Trustee message (first user if role=1, first assistant if role=2)
        investor_msg = None
        trustee_msg = None
        for msg in messages:
            if role == 1:  # We are Investor
                if msg["role"] == "assistant" and investor_msg is None:
                    investor_msg = msg["content"]
                elif msg["role"] == "user" and trustee_msg is None:
                    trustee_msg = msg["content"]
            else:  # We are Trustee (role=2)
                if msg["role"] == "user" and investor_msg is None:
                    investor_msg = msg["content"]
                elif msg["role"] == "assistant" and trustee_msg is None:
                    trustee_msg = msg["content"]
        
        # Extract numbers
        send_amount = extract_number(investor_msg, 0, endowment) if investor_msg else None
        pool = multiplier * send_amount if send_amount is not None else 0
        return_amount = extract_number(trustee_msg, 0, pool) if trustee_msg else None
        
        if send_amount is None:
            send_amount = 0  # Investor failed to state amount → sends nothing
        if return_amount is None:
            return_amount = 0  # Trustee failed to state amount → returns nothing
        
        # Compute payoffs
        investor_payoff = (endowment - send_amount) + return_amount
        trustee_payoff = (multiplier * send_amount) - return_amount
        
        # R_coop
        U_A = investor_payoff if role == 1 else trustee_payoff
        U_B = trustee_payoff if role == 1 else investor_payoff
        
        R_coop = (
            lambda_self * U_A
            + lambda_welfare * (U_A + U_B)
            + lambda_fair * (U_A * U_B) / 100.0
        )
        
        rewards.append(R_coop)
    
    return rewards
```

## Dialogue Flow

The Trust Game is **sequential** (not simultaneous), so it maps directly to the existing
multi-turn dialogue without any architectural changes:

```
Turn 1: Investor (assistant) → "I send X points"
Turn 2: Trustee (user)       → "I return Y points"
[Optional Turn 3-4: brief follow-up, but the key actions are in turns 1-2]
```

The trainer's `_play_negotiation()` already handles this alternating pattern.
Set `max_rounds=2` for this game (1 exchange = 2 messages is sufficient).

For the opponent (frozen copy or GPT-4o-mini), the system prompt is prompt_2.
The agent's system prompt is prompt (always index 0 in the messages list).

## WandB Metrics to Log

Inside the reward function, add wandb logging similar to NegotiationEnv:

```python
# Per-batch metrics:
"gt/trust/avg_send"           # Average amount sent by Investor
"gt/trust/avg_return"         # Average amount returned by Trustee
"gt/trust/avg_return_ratio"   # return / (3 × send), 0 if send=0
"gt/trust/deal_rate"          # % of games where send > 0
"gt/trust/fair_rate"          # % of games where Trustee returns ≥ 40% of pool
"gt/trust/U_A_mean"           # Mean agent payoff
"gt/trust/U_B_mean"           # Mean opponent payoff
"gt/trust/social_welfare"     # Mean U_A + U_B
"gt/trust/R_coop_mean"        # Mean cooperative reward
```

## Implementation Order

1. Create `envs/game_theory/__init__.py` (empty)
2. Create `envs/game_theory/prompts.py` (just the prompt strings)
3. Create `envs/game_theory/action_extractor.py` (regex extraction)
4. Create `envs/game_theory/env.py` (GameTheoryEnv class)
5. Create `tests/test_trust_game.py` (verify everything works)
6. Modify `grpo_single_gpu.py` (add factory function)
7. Test end-to-end: `python multiturn_llm_training/grpo/grpo_single_gpu.py --game-type trust-game --test`

## Important Constraints

- The GameTheoryEnv must produce datasets with EXACTLY the same field names as NegotiationEnv,
  because the GRPOTrainer reads these fields by name.
- The reward function signature must match what the trainer expects.
- Do NOT touch the existing NegotiationEnv or negotiation configs.
- Do NOT use GPT-4o-mini for action extraction — use regex only.
- The `game_config` dict must be JSON-serializable (no numpy arrays, no classes).
- Keep it simple. The Trust Game is a 2-turn game. Don't over-engineer.
- When in doubt about how something works, look at how NegotiationEnv does it.

## Test Commands

```bash
# Quick smoke test (verifies dataset creation + reward computation)
python tests/test_trust_game.py

# Training test (2 generations, 2 rounds, 100 samples — should complete in <5 min)
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --game-type trust-game \
    --test

# Full training run
python multiturn_llm_training/grpo/grpo_single_gpu.py \
    --game-type trust-game \
    --lambda-self 1.0 \
    --lambda-welfare 0.5 \
    --lambda-fair 0.3 \
    --num-generations 8 \
    --train-size 200 \
    --max-rounds 2 \
    --use-wandb \
    --run-name trust_game_cooperative_v1
```

## R_coop Analysis for the Trust Game

This is why the Trust Game is the best showcase for R_coop.

With pure self-interest (λ_self=1, λ_welfare=0, λ_fair=0), the Trustee agent learns:
"Keep everything → my payoff is maximized." The Investor learns: "Don't send anything."
Result: Nash equilibrium (10, 0). Total surplus: 10.

With cooperative reward (needs sufficient λ_fair), the Trustee learns:
"Return a fair share → high Nash product → high R_coop." The Investor learns: "Send everything → 
trust is rewarded." Result: approaches Pareto optimum (15, 15). Total surplus: 30.

The exact λ_fair threshold where behavior flips from selfish to fair is an empirical question
that the training experiments will answer. That's a key result for the thesis.
