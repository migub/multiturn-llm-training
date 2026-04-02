"""
Trust Game Evaluation Script

Standalone script to evaluate trained checkpoints on the Trust Game.
Tests whether cooperative training generalizes beyond negotiation.

Trust Game:
  - Investor has 10 points, sends 0-10 to Trustee
  - Sent amount is tripled (x3)
  - Trustee returns 0 to 3*sent back to Investor
  - Nash equilibrium (selfish): send 0, return 0 -> (10, 0)
  - Pareto optimum: send 10, return 15 -> (15, 15)

Usage:
  python evaluations/trust_game_eval.py --checkpoint output/run_x/checkpoint-100 --num-games 50
  python evaluations/trust_game_eval.py --checkpoint none --num-games 10  # base model baseline
"""

import argparse
import json
import os
import re
import sys
import time
from statistics import mean, stdev

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


# ============================================================
# System Prompts
# ============================================================

INVESTOR_PROMPT = """You are playing the Trust Game as the Investor.

Rules:
- You start with 10 points.
- You choose how many points (0-10, whole numbers only) to send to the Trustee.
- Any points you send are TRIPLED before the Trustee receives them.
- The Trustee then decides how many points to return to you.
- Your final payoff = (10 - amount sent) + amount returned by Trustee.

Example: If you send 6 points, you keep 4. The Trustee receives 18 (6x3). If the Trustee returns 9, your payoff = 4 + 9 = 13. Trustee's payoff = 18 - 9 = 9.

You must decide how many points to send. State your decision clearly.
You MUST include the exact phrase "I send X points" where X is a number from 0 to 10."""

TRUSTEE_PROMPT = """You are playing the Trust Game as the Trustee.

Rules:
- The Investor started with 10 points and chose to send some to you.
- Whatever they sent was TRIPLED — that is your total received amount.
- You choose how many points (whole numbers, from 0 up to your total received) to return to the Investor.
- Your final payoff = total received - amount returned.
- Investor's final payoff = (10 - amount they sent) + amount you return.

Example: If the Investor sends 6, you receive 18 (6x3). If you return 9, your payoff = 18 - 9 = 9. Investor's payoff = 4 + 9 = 13.

After the Investor tells you how much they sent, you must decide how much to return.
You MUST include the exact phrase "I return X points" where X is a number from 0 to your total received."""


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
        text, return_tensors="pt", truncation=True, max_length=1800
    ).to(model.device)

    outputs = model.generate(**inputs, generation_config=gen_config)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return strip_think_tags(response)


# ============================================================
# Action Extraction
# ============================================================

def extract_send_amount(text):
    """Extract the amount sent by the Investor. Returns int or None."""
    if not text:
        return None

    patterns = [
        r"I send (\d+) points?",
        r"send (\d+) points?",
        r"sending (\d+) points?",
        r"I(?:'ll| will) send (\d+)",
        r"decide to send (\d+)",
        r"SEND:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            return max(0, min(10, val))

    # Fallback: last number in range 0-10
    numbers = [int(n) for n in re.findall(r"\b(\d+)\b", text) if 0 <= int(n) <= 10]
    if numbers:
        return numbers[-1]

    return None


def extract_return_amount(text, max_return):
    """Extract the amount returned by the Trustee. Returns int or None."""
    if not text or max_return == 0:
        return None

    patterns = [
        r"I return (\d+) points?",
        r"return (\d+) points?",
        r"returning (\d+) points?",
        r"I(?:'ll| will) return (\d+)",
        r"send(?:ing)? back (\d+)",
        r"give(?:ing)? back (\d+)",
        r"RETURN:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            return max(0, min(max_return, val))

    # Fallback: last number in range 0-max_return
    numbers = [int(n) for n in re.findall(r"\b(\d+)\b", text) if 0 <= int(n) <= max_return]
    if numbers:
        return numbers[-1]

    return None


# ============================================================
# Game Play
# ============================================================

def has_adapter(model):
    """Check if model is a PeftModel with adapters."""
    return hasattr(model, "enable_adapter_layers")


def play_single_game(model, tokenizer, gen_config, trained_role):
    """
    Play one Trust Game. The trained model plays `trained_role` (investor/trustee),
    the frozen base model plays the other role.

    Returns dict with investor_text, trustee_text, send, return_amt, parse_ok.
    """
    # Turn 1: Investor decides how much to send
    investor_messages = [{"role": "system", "content": INVESTOR_PROMPT}]

    if has_adapter(model):
        if trained_role == "investor":
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()

    investor_response = gen_response(model, tokenizer, gen_config, investor_messages)

    # Extract send amount before Trustee responds (needed for Trustee prompt context)
    send = extract_send_amount(investor_response)

    # Turn 2: Trustee decides how much to return
    trustee_messages = [
        {"role": "system", "content": TRUSTEE_PROMPT},
        {"role": "user", "content": investor_response},
    ]

    if has_adapter(model):
        if trained_role == "trustee":
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()

    trustee_response = gen_response(model, tokenizer, gen_config, trustee_messages)

    # Re-enable adapters (clean state)
    if has_adapter(model):
        model.enable_adapter_layers()

    # Extract return amount
    max_return = (send * 3) if send is not None else 0
    return_amt = extract_return_amount(trustee_response, max_return)

    return {
        "investor_text": investor_response,
        "trustee_text": trustee_response,
        "send": send,
        "return_amt": return_amt,
        "parse_ok": send is not None and return_amt is not None,
    }


def play_trust_games(model, tokenizer, gen_config, num_games, verbose=False):
    """Play N trust games per role. Returns dict with 'investor' and 'trustee' game lists."""
    results = {"investor": [], "trustee": []}

    for role in ["investor", "trustee"]:
        print(f"\nPlaying {num_games} games as {role.upper()}...")
        for i in range(num_games):
            t0 = time.time()
            game = play_single_game(model, tokenizer, gen_config, trained_role=role)
            elapsed = time.time() - t0

            status = "OK" if game["parse_ok"] else "PARSE FAIL"
            send_str = str(game["send"]) if game["send"] is not None else "?"
            ret_str = str(game["return_amt"]) if game["return_amt"] is not None else "?"
            print(f"  Game {i+1:3d}/{num_games}: send={send_str}, return={ret_str}  [{status}] ({elapsed:.1f}s)")

            if verbose:
                inv_payoff = ((10 - game["send"]) + game["return_amt"]) if game["parse_ok"] else "?"
                tru_payoff = ((game["send"] * 3) - game["return_amt"]) if game["parse_ok"] else "?"
                print(f"    Investor: {game['investor_text'][:200]}")
                print(f"    Trustee:  {game['trustee_text'][:200]}")
                print(f"    Payoffs:  Investor={inv_payoff}, Trustee={tru_payoff}")
                print()

            results[role].append(game)

    return results


# ============================================================
# Metrics
# ============================================================

def compute_metrics(games):
    """Compute aggregate metrics for a list of games (one role)."""
    valid = [g for g in games if g["parse_ok"]]
    if not valid:
        return None

    sends = [g["send"] for g in valid]
    returns = [g["return_amt"] for g in valid]
    inv_payoffs = [(10 - g["send"]) + g["return_amt"] for g in valid]
    tru_payoffs = [(g["send"] * 3) - g["return_amt"] for g in valid]
    social_welfare = [ip + tp for ip, tp in zip(inv_payoffs, tru_payoffs)]
    nash_product = [ip * tp for ip, tp in zip(inv_payoffs, tru_payoffs)]
    return_ratios = [
        g["return_amt"] / (g["send"] * 3) if g["send"] > 0 else 0.0
        for g in valid
    ]

    n = len(valid)
    return {
        "n_valid": n,
        "n_total": len(games),
        "parse_failure_rate": 1 - n / len(games),
        "avg_send": mean(sends),
        "avg_return": mean(returns),
        "avg_return_ratio": mean(return_ratios),
        "avg_investor_payoff": mean(inv_payoffs),
        "avg_trustee_payoff": mean(tru_payoffs),
        "avg_social_welfare": mean(social_welfare),
        "avg_nash_product": mean(nash_product),
        "std_send": stdev(sends) if n > 1 else 0.0,
        "std_return": stdev(returns) if n > 1 else 0.0,
        "std_investor_payoff": stdev(inv_payoffs) if n > 1 else 0.0,
        "std_trustee_payoff": stdev(tru_payoffs) if n > 1 else 0.0,
        "std_social_welfare": stdev(social_welfare) if n > 1 else 0.0,
        "std_nash_product": stdev(nash_product) if n > 1 else 0.0,
    }


# ============================================================
# Reporting
# ============================================================

def report_results(all_metrics, results, args):
    """Print formatted results to console and optionally save JSON."""
    print("\n" + "=" * 60)
    print("TRUST GAME EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Model:         {args.model_name}")
    print(f"  Games/role:    {args.num_games}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Seed:          {args.seed}")
    print("-" * 60)
    print(f"  Reference — Nash equilibrium:  send=0,  return=0   -> (10, 0)  SW=10  NP=0")
    print(f"  Reference — Pareto optimum:    send=10, return=15  -> (15, 15) SW=30  NP=225")
    print("-" * 60)

    for role in ["investor", "trustee"]:
        m = all_metrics[role]
        if m is None:
            print(f"\n  [{role.upper()}] All games failed to parse.")
            continue

        print(f"\n  [Trained as {role.upper()}]  (valid: {m['n_valid']}/{m['n_total']})")
        print(f"    Avg send:            {m['avg_send']:5.2f} / 10    (std: {m['std_send']:.2f})")
        print(f"    Avg return:          {m['avg_return']:5.2f}          (std: {m['std_return']:.2f})")
        print(f"    Return ratio:        {m['avg_return_ratio']:5.1%}")
        print(f"    Investor payoff:     {m['avg_investor_payoff']:5.2f}          (std: {m['std_investor_payoff']:.2f})")
        print(f"    Trustee payoff:      {m['avg_trustee_payoff']:5.2f}          (std: {m['std_trustee_payoff']:.2f})")
        print(f"    Social welfare:      {m['avg_social_welfare']:5.2f}          (Nash: 10, Pareto: 30)")
        print(f"    Nash product:        {m['avg_nash_product']:5.1f}           (Nash: 0, Pareto: 225)")
        print(f"    Parse failure rate:  {m['parse_failure_rate']:5.1%}")

    print("=" * 60)

    if args.output_json:
        output = {
            "args": {
                "checkpoint": args.checkpoint,
                "model_name": args.model_name,
                "num_games": args.num_games,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "metrics": all_metrics,
            "games": {
                role: [
                    {
                        "investor_text": g["investor_text"],
                        "trustee_text": g["trustee_text"],
                        "send": g["send"],
                        "return_amt": g["return_amt"],
                        "parse_ok": g["parse_ok"],
                    }
                    for g in games
                ]
                for role, games in results.items()
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
    parser = argparse.ArgumentParser(description="Trust Game evaluation for trained checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint dir, or 'none' for base model")
    parser.add_argument("--model-name", type=str, default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument("--num-games", type=int, default=50,
                        help="Number of games per role (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--quantized", action="store_true", default=True)
    parser.add_argument("--no-quantized", action="store_false", dest="quantized")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save full results as JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full dialogues and payoffs for each game")
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
    results = play_trust_games(model, tokenizer, gen_config, args.num_games, verbose=args.verbose)

    # Compute metrics
    all_metrics = {
        "investor": compute_metrics(results["investor"]),
        "trustee": compute_metrics(results["trustee"]),
    }

    # Report
    report_results(all_metrics, results, args)


if __name__ == "__main__":
    main()
