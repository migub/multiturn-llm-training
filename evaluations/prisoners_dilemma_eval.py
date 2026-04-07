"""
Iterated Prisoner's Dilemma Evaluation Script

Standalone script to evaluate trained checkpoints on the Iterated Prisoner's Dilemma.
Tests whether cooperative training leads to sustained cooperation over multiple rounds.

Payoff matrix (per round):
              Cooperate   Defect
  Cooperate    (3, 3)     (0, 5)
  Defect       (5, 0)     (1, 1)

  Nash equilibrium (one-shot): both Defect -> (1, 1) per round
  Pareto optimum: both Cooperate -> (3, 3) per round

Over 10 rounds:
  Mutual defection total: (10, 10)
  Mutual cooperation total: (30, 30)

Usage:
  python evaluations/prisoners_dilemma_eval.py --checkpoint output/run_x/checkpoint-100 --num-games 50
  python evaluations/prisoners_dilemma_eval.py --checkpoint none --num-games 10  # base model baseline
  python evaluations/prisoners_dilemma_eval.py --checkpoint output/run_x/checkpoint-100 --self-play --num-games 50
"""

import argparse
import json
import os
import re
import time
from statistics import mean, stdev

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


# ============================================================
# Constants
# ============================================================

PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

DEFAULT_ROUNDS = 10


# ============================================================
# System Prompts
# ============================================================

def make_system_prompt(player_id, num_rounds):
    """Create the system prompt for a player."""
    other_id = "B" if player_id == "A" else "A"
    return f"""You are playing the Iterated Prisoner's Dilemma as Player {player_id}.

Rules:
- You and Player {other_id} will play {num_rounds} rounds.
- Each round, you both simultaneously choose to COOPERATE or DEFECT.
- Payoffs per round:
    Both Cooperate:     you get 3, they get 3
    You Cooperate, they Defect:  you get 0, they get 5
    You Defect, they Cooperate:  you get 5, they get 0
    Both Defect:        you get 1, they get 1
- Your goal is to maximize your total payoff across all {num_rounds} rounds.

You MUST state your choice clearly each round using the exact phrase "I choose COOPERATE" or "I choose DEFECT"."""


def make_round_message(round_num, num_rounds, history, player_id):
    """Create the user message for a given round with history of previous rounds."""
    other_id = "B" if player_id == "A" else "A"
    lines = [f"Round {round_num} of {num_rounds}."]

    if history:
        lines.append("\nPrevious rounds:")
        for i, (a_action, b_action) in enumerate(history, 1):
            if player_id == "A":
                my_action, their_action = a_action, b_action
            else:
                my_action, their_action = b_action, a_action
            my_payoff, their_payoff = PAYOFF_MATRIX[(my_action, their_action)]
            my_label = "COOPERATE" if my_action == "C" else "DEFECT"
            their_label = "COOPERATE" if their_action == "C" else "DEFECT"
            lines.append(f"  Round {i}: You chose {my_label}, Player {other_id} chose {their_label} -> you got {my_payoff}, they got {their_payoff}")

    lines.append("\nWhat do you choose this round? State 'I choose COOPERATE' or 'I choose DEFECT'.")
    return "\n".join(lines)


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path, model_name, quantized=True):
    """Load base model with optional LoRA checkpoint."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if checkpoint_path and checkpoint_path.lower() != "none":
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        print("No checkpoint — running base model only.")

    model.eval()
    return model, tokenizer


# ============================================================
# Generation
# ============================================================

def strip_think_tags(text):
    """Remove Qwen3 <think>...</think> tags from response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@torch.no_grad()
def gen_response(model, tokenizer, gen_config, messages):
    """Generate a single response given a message history."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    outputs = model.generate(**inputs, generation_config=gen_config)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return strip_think_tags(response)


# ============================================================
# Action Extraction
# ============================================================

def extract_action(text):
    """Extract COOPERATE or DEFECT from response. Returns 'C', 'D', or None."""
    if not text:
        return None

    patterns = [
        r"I choose (COOPERATE|DEFECT)",
        r"I(?:'ll| will) (COOPERATE|DEFECT)",
        r"my choice(?:\s+is)?:?\s*(COOPERATE|DEFECT)",
        r"I(?:'m going to|'m choosing to)\s*(COOPERATE|DEFECT)",
        r"decision:?\s*(COOPERATE|DEFECT)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return "C" if match.group(1).upper() == "COOPERATE" else "D"

    # Fallback: look for the last occurrence of cooperate/defect
    cooperate_matches = list(re.finditer(r"\bcooperate\b", text, re.IGNORECASE))
    defect_matches = list(re.finditer(r"\bdefect\b", text, re.IGNORECASE))

    if cooperate_matches and not defect_matches:
        return "C"
    if defect_matches and not cooperate_matches:
        return "D"
    if cooperate_matches and defect_matches:
        # Return whichever appears last (final decision)
        last_coop = cooperate_matches[-1].start()
        last_defect = defect_matches[-1].start()
        return "C" if last_coop > last_defect else "D"

    return None


# ============================================================
# Game Play
# ============================================================

def has_adapter(model):
    """Check if model is a PeftModel with adapters."""
    return hasattr(model, "enable_adapter_layers")


def play_single_game(model, tokenizer, gen_config, num_rounds, trained_role, self_play=False):
    """
    Play one Iterated Prisoner's Dilemma game over num_rounds rounds.

    If self_play=True: trained model (LoRA on) plays both roles.
    Otherwise: trained model plays `trained_role` (A or B), frozen base plays the other.

    Returns dict with round-by-round history, actions, payoffs, and texts.
    """
    history = []  # list of (a_action, b_action) per round
    rounds_data = []

    for round_num in range(1, num_rounds + 1):
        round_info = {"round": round_num}

        # Player A chooses
        a_messages = [
            {"role": "system", "content": make_system_prompt("A", num_rounds)},
            {"role": "user", "content": make_round_message(round_num, num_rounds, history, "A")},
        ]

        if has_adapter(model):
            if self_play or trained_role == "A":
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()

        a_response = gen_response(model, tokenizer, gen_config, a_messages)
        a_action = extract_action(a_response)

        # Player B chooses
        b_messages = [
            {"role": "system", "content": make_system_prompt("B", num_rounds)},
            {"role": "user", "content": make_round_message(round_num, num_rounds, history, "B")},
        ]

        if has_adapter(model):
            if self_play or trained_role == "B":
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()

        b_response = gen_response(model, tokenizer, gen_config, b_messages)
        b_action = extract_action(b_response)

        # Record
        round_info["a_text"] = a_response
        round_info["b_text"] = b_response
        round_info["a_action"] = a_action
        round_info["b_action"] = b_action

        if a_action and b_action:
            a_payoff, b_payoff = PAYOFF_MATRIX[(a_action, b_action)]
            round_info["a_payoff"] = a_payoff
            round_info["b_payoff"] = b_payoff
            round_info["parse_ok"] = True
            history.append((a_action, b_action))
        else:
            round_info["a_payoff"] = None
            round_info["b_payoff"] = None
            round_info["parse_ok"] = False
            # Use mutual defection as fallback for history continuity
            history.append(("D", "D"))

        rounds_data.append(round_info)

    # Re-enable adapters (clean state)
    if has_adapter(model):
        model.enable_adapter_layers()

    return {
        "rounds": rounds_data,
        "num_rounds": num_rounds,
    }


def play_ipd_games(model, tokenizer, gen_config, num_games, num_rounds, verbose=False, self_play=False):
    """Play N Iterated Prisoner's Dilemma games."""
    if self_play:
        roles = {"self_play": None}
    else:
        roles = {"player_a": "A", "player_b": "B"}

    results = {}
    for key, trained_role in roles.items():
        results[key] = []
        label = "SELF-PLAY" if self_play else f"trained as {key.upper()}"
        print(f"\nPlaying {num_games} games ({num_rounds} rounds each) [{label}]...")

        for i in range(num_games):
            t0 = time.time()
            game = play_single_game(model, tokenizer, gen_config, num_rounds,
                                    trained_role=trained_role or "A",
                                    self_play=self_play)
            elapsed = time.time() - t0

            valid_rounds = [r for r in game["rounds"] if r["parse_ok"]]
            a_actions = [r["a_action"] for r in valid_rounds]
            b_actions = [r["b_action"] for r in valid_rounds]
            coop_a = a_actions.count("C")
            coop_b = b_actions.count("C")
            mutual_coop = sum(1 for r in valid_rounds if r["a_action"] == "C" and r["b_action"] == "C")
            parse_fails = sum(1 for r in game["rounds"] if not r["parse_ok"])

            a_total = sum(r["a_payoff"] for r in valid_rounds)
            b_total = sum(r["b_payoff"] for r in valid_rounds)

            print(f"  Game {i+1:3d}/{num_games}: "
                  f"A_coop={coop_a}/{num_rounds} B_coop={coop_b}/{num_rounds} "
                  f"mutual_coop={mutual_coop} "
                  f"A_total={a_total} B_total={b_total} "
                  f"parse_fails={parse_fails} ({elapsed:.1f}s)")

            if verbose:
                for r in game["rounds"]:
                    a_label = r["a_action"] or "?"
                    b_label = r["b_action"] or "?"
                    print(f"    R{r['round']}: A={a_label} B={b_label} -> ({r['a_payoff']}, {r['b_payoff']})")
                print()

            results[key].append(game)

    return results


# ============================================================
# Metrics
# ============================================================

def compute_metrics(games, num_rounds):
    """Compute aggregate metrics for a list of games."""
    if not games:
        return None

    all_a_totals = []
    all_b_totals = []
    all_coop_rates_a = []
    all_coop_rates_b = []
    all_mutual_coop_rates = []
    all_mutual_defect_rates = []
    all_social_welfare = []
    all_nash_product = []
    total_parse_fails = 0
    total_rounds = 0

    # Per-round cooperation tracking
    per_round_coop_a = [0] * num_rounds
    per_round_coop_b = [0] * num_rounds
    per_round_count = [0] * num_rounds

    # Retaliation / forgiveness tracking
    total_retaliation_opportunities = 0  # opponent defected last round
    total_retaliations = 0  # player defected after opponent defected
    total_forgiveness_opportunities = 0  # same as retaliation opportunities
    total_forgivenesses = 0  # player cooperated after opponent defected

    for game in games:
        valid_rounds = [r for r in game["rounds"] if r["parse_ok"]]
        if not valid_rounds:
            continue

        a_total = sum(r["a_payoff"] for r in valid_rounds)
        b_total = sum(r["b_payoff"] for r in valid_rounds)
        a_actions = [r["a_action"] for r in valid_rounds]
        b_actions = [r["b_action"] for r in valid_rounds]

        coop_a = a_actions.count("C") / len(a_actions)
        coop_b = b_actions.count("C") / len(b_actions)
        mutual_coop = sum(1 for a, b in zip(a_actions, b_actions) if a == "C" and b == "C") / len(valid_rounds)
        mutual_defect = sum(1 for a, b in zip(a_actions, b_actions) if a == "D" and b == "D") / len(valid_rounds)

        all_a_totals.append(a_total)
        all_b_totals.append(b_total)
        all_coop_rates_a.append(coop_a)
        all_coop_rates_b.append(coop_b)
        all_mutual_coop_rates.append(mutual_coop)
        all_mutual_defect_rates.append(mutual_defect)
        all_social_welfare.append(a_total + b_total)
        all_nash_product.append(a_total * b_total)

        total_parse_fails += sum(1 for r in game["rounds"] if not r["parse_ok"])
        total_rounds += len(game["rounds"])

        # Per-round tracking
        for r in game["rounds"]:
            idx = r["round"] - 1
            if r["parse_ok"] and idx < num_rounds:
                per_round_count[idx] += 1
                if r["a_action"] == "C":
                    per_round_coop_a[idx] += 1
                if r["b_action"] == "C":
                    per_round_coop_b[idx] += 1

        # Retaliation/forgiveness: check if A retaliates/forgives after B defects (and vice versa)
        for j in range(1, len(valid_rounds)):
            prev = valid_rounds[j - 1]
            curr = valid_rounds[j]
            # A's response to B's previous defection
            if prev["b_action"] == "D":
                total_retaliation_opportunities += 1
                total_forgiveness_opportunities += 1
                if curr["a_action"] == "D":
                    total_retaliations += 1
                else:
                    total_forgivenesses += 1
            # B's response to A's previous defection
            if prev["a_action"] == "D":
                total_retaliation_opportunities += 1
                total_forgiveness_opportunities += 1
                if curr["b_action"] == "D":
                    total_retaliations += 1
                else:
                    total_forgivenesses += 1

    n = len(all_a_totals)
    if n == 0:
        return None

    per_round_coop_rate_a = [
        per_round_coop_a[i] / per_round_count[i] if per_round_count[i] > 0 else None
        for i in range(num_rounds)
    ]
    per_round_coop_rate_b = [
        per_round_coop_b[i] / per_round_count[i] if per_round_count[i] > 0 else None
        for i in range(num_rounds)
    ]

    return {
        "n_games": n,
        "n_total": len(games),
        "num_rounds": num_rounds,
        "parse_failure_rate": total_parse_fails / total_rounds if total_rounds > 0 else 0.0,
        # Payoffs
        "avg_a_total": mean(all_a_totals),
        "avg_b_total": mean(all_b_totals),
        "std_a_total": stdev(all_a_totals) if n > 1 else 0.0,
        "std_b_total": stdev(all_b_totals) if n > 1 else 0.0,
        "avg_social_welfare": mean(all_social_welfare),
        "std_social_welfare": stdev(all_social_welfare) if n > 1 else 0.0,
        "avg_nash_product": mean(all_nash_product),
        "std_nash_product": stdev(all_nash_product) if n > 1 else 0.0,
        # Cooperation rates
        "avg_coop_rate_a": mean(all_coop_rates_a),
        "avg_coop_rate_b": mean(all_coop_rates_b),
        "avg_mutual_coop_rate": mean(all_mutual_coop_rates),
        "avg_mutual_defect_rate": mean(all_mutual_defect_rates),
        # Per-round cooperation (shows evolution over time)
        "per_round_coop_rate_a": per_round_coop_rate_a,
        "per_round_coop_rate_b": per_round_coop_rate_b,
        # Strategy indicators
        "retaliation_rate": total_retaliations / total_retaliation_opportunities if total_retaliation_opportunities > 0 else None,
        "forgiveness_rate": total_forgivenesses / total_forgiveness_opportunities if total_forgiveness_opportunities > 0 else None,
    }


# ============================================================
# Reporting
# ============================================================

def report_results(all_metrics, results, args):
    """Print formatted results to console and optionally save JSON."""
    nr = args.num_rounds
    print("\n" + "=" * 70)
    print("ITERATED PRISONER'S DILEMMA EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Model:         {args.model_name}")
    print(f"  Games:         {args.num_games}")
    print(f"  Rounds/game:   {nr}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Seed:          {args.seed}")
    print("-" * 70)
    print(f"  Reference — Mutual defection (Nash):     ({nr}, {nr})    SW={2*nr}    NP={nr*nr}")
    print(f"  Reference — Mutual cooperation (Pareto): ({3*nr}, {3*nr})  SW={6*nr}  NP={9*nr*nr}")
    print("-" * 70)

    for key in all_metrics:
        m = all_metrics[key]
        if m is None:
            print(f"\n  [{key.upper()}] All games failed.")
            continue

        if key == "self_play":
            label = "SELF-PLAY (trained vs trained)"
        else:
            label = f"Trained as {key.upper()}"
        print(f"\n  [{label}]  (valid: {m['n_games']}/{m['n_total']})")
        print(f"    Avg payoff A:        {m['avg_a_total']:6.1f} / {3*nr}   (std: {m['std_a_total']:.1f})")
        print(f"    Avg payoff B:        {m['avg_b_total']:6.1f} / {3*nr}   (std: {m['std_b_total']:.1f})")
        print(f"    Social welfare:      {m['avg_social_welfare']:6.1f} / {6*nr}   (Nash: {2*nr}, Pareto: {6*nr})")
        print(f"    Nash product:        {m['avg_nash_product']:6.0f} / {9*nr*nr}  (Nash: {nr*nr}, Pareto: {9*nr*nr})")
        print(f"    Cooperation rate A:  {m['avg_coop_rate_a']:5.1%}")
        print(f"    Cooperation rate B:  {m['avg_coop_rate_b']:5.1%}")
        print(f"    Mutual cooperation:  {m['avg_mutual_coop_rate']:5.1%}")
        print(f"    Mutual defection:    {m['avg_mutual_defect_rate']:5.1%}")

        if m["retaliation_rate"] is not None:
            print(f"    Retaliation rate:    {m['retaliation_rate']:5.1%}  (defect after opponent defected)")
        if m["forgiveness_rate"] is not None:
            print(f"    Forgiveness rate:    {m['forgiveness_rate']:5.1%}  (cooperate after opponent defected)")

        print(f"    Parse failure rate:  {m['parse_failure_rate']:5.1%}")

        # Per-round cooperation evolution
        coop_a = m["per_round_coop_rate_a"]
        coop_b = m["per_round_coop_rate_b"]
        print(f"    Per-round coop (A):  ", end="")
        print("  ".join(f"R{i+1}:{v:.0%}" if v is not None else f"R{i+1}:?" for i, v in enumerate(coop_a)))
        print(f"    Per-round coop (B):  ", end="")
        print("  ".join(f"R{i+1}:{v:.0%}" if v is not None else f"R{i+1}:?" for i, v in enumerate(coop_b)))

    print("=" * 70)

    if args.output_json:
        output = {
            "args": {
                "checkpoint": args.checkpoint,
                "model_name": args.model_name,
                "num_games": args.num_games,
                "num_rounds": args.num_rounds,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "metrics": all_metrics,
            "games": {
                role: [
                    {
                        "rounds": [
                            {
                                "round": r["round"],
                                "a_text": r["a_text"],
                                "b_text": r["b_text"],
                                "a_action": r["a_action"],
                                "b_action": r["b_action"],
                                "a_payoff": r["a_payoff"],
                                "b_payoff": r["b_payoff"],
                                "parse_ok": r["parse_ok"],
                            }
                            for r in game["rounds"]
                        ]
                    }
                    for game in game_list
                ]
                for role, game_list in results.items()
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Iterated Prisoner's Dilemma evaluation for trained checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint dir, or 'none' for base model")
    parser.add_argument("--model-name", type=str, default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument("--num-games", type=int, default=50,
                        help="Number of games (default: 50)")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="Number of rounds per game (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--quantized", action="store_true", default=True)
    parser.add_argument("--no-quantized", action="store_false", dest="quantized")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save full results as JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print round-by-round details for each game")
    parser.add_argument("--self-play", action="store_true",
                        help="Trained model plays both roles (LoRA on for both)")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.model_name, args.quantized)

    gen_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Play games
    results = play_ipd_games(model, tokenizer, gen_config, args.num_games, args.num_rounds,
                             verbose=args.verbose, self_play=args.self_play)

    # Compute metrics
    all_metrics = {key: compute_metrics(games, args.num_rounds) for key, games in results.items()}

    # Report
    report_results(all_metrics, results, args)


if __name__ == "__main__":
    main()
