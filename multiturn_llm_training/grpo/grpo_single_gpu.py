"""
Single-GPU Multi-Turn GRPO Training for Negotiation.

Usage:
    python multiturn_llm_training/grpo/train.py
    python multiturn_llm_training/grpo/train.py --game-type multi-game --num-train-steps 400
    python multiturn_llm_training/grpo/train.py --test  # Quick test with minimal settings
"""

import sys
import os
import argparse
import torch

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from envs.negotiation.env import NegotiationEnv
from trl.trainer.grpo_trainer_multiturn import GRPOTrainer
from trl import GRPOConfig
from peft import LoraConfig
from unsloth import FastLanguageModel


def main(args):
    print("=" * 60)
    print("Multi-Turn GRPO Negotiation Training (Single GPU)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Game type: {args.game_type}")
    print(f"Group size (G): {args.num_generations}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Train size: {args.train_size}")
    print(f"Lambda self: {args.lambda_self}")
    print(f"Lambda welfare: {args.lambda_welfare}")
    print(f"Lambda fair: {args.lambda_fair}")
    print()

    # Ensure deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load Model with Unsloth ----
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for GRPO

    print(f"Model loaded.")
    model.print_trainable_parameters()

    # ---- Setup Environment ----
    print(f"\nSetting up environment: {args.game_type}")
    negotiation_env = NegotiationEnv(game_type=args.game_type, lambda_self=args.lambda_self, lambda_welfare=args.lambda_welfare, lambda_fair=args.lambda_fair)
    train_dataset = negotiation_env.create_dataset(size=args.train_size)
    eval_dataset = negotiation_env.create_dataset(size=args.eval_size)
    reward_functions = negotiation_env.get_reward_functions()

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset: {len(eval_dataset)} samples")

    # ---- Training Config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        num_iterations=1,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.num_generations,
        per_device_eval_batch_size=args.num_generations,
        num_generations=args.num_generations,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=True,
        use_vllm=False,
        logging_steps=args.logging_steps,
        log_completions=True,
        report_to="wandb" if args.use_wandb else "none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=False,
        beta=args.beta,
        scale_rewards="group",
        loss_type="grpo",
    )

    print(f"\nTraining config:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  Num generations (G): {args.num_generations}")
    print(f"  Max negotiation rounds: {args.max_rounds}")
    print(f"  Max tokens per turn: {args.max_tokens_per_turn}")

    # ---- Create Trainer ----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        multiturn=True,
        max_negotiation_rounds=args.max_rounds,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )

    # ---- Train ----
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # ---- Save ----
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Model saved to {args.output_dir}/final")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Turn GRPO Negotiation Training")

    # Model
    parser.add_argument("--model-name", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)  

    # Environment
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--lambda-self", type=float, default=1.0)
    parser.add_argument("--lambda-welfare", type=float, default=0.5)
    parser.add_argument("--lambda-fair", type=float, default=0.3)

    # Training
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-tokens-per-turn", type=int, default=200)
    parser.add_argument("--max-completion-length", type=int, default=2048)

    # Logging & Saving
    parser.add_argument("--output-dir", type=str, default="output/grpo_multiturn")
    parser.add_argument("--run-name", type=str, default="grpo_multiturn_rental")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--use-wandb", action="store_true", default=False)

    # Quick test mode
    parser.add_argument("--test", action="store_true", default=False,
                        help="Quick test with minimal settings")

    args = parser.parse_args()

    # Override for quick test
    if args.test:
        args.train_size = 100
        args.eval_size = 10
        args.num_generations = 2
        args.max_rounds = 2
        args.save_steps = 999
        args.eval_steps = 999
        args.logging_steps = 1
        print("*** TEST MODE: minimal settings ***\n")

    main(args)