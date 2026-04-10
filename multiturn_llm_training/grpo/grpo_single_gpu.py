import sys 
import os 
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from envs.negotiation.env import NegotiationEnv 
from multiturn_llm_training.grpo.multiturn_grpo_trainer import MultiTurnGRPOTrainer
from trl import GRPOConfig
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):

    print("Training Args:\n", args)

    # Ensure deterministic behaviour across runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    os.makedirs(f"{args.output_dir}/{args.run_name}", exist_ok=True)

    # ---- Setup Environment ----
    negotiation_env = NegotiationEnv(
        game_type=args.game_type,
        lambda_self=args.lambda_self,
        lambda_welfare=args.lambda_welfare,
        lambda_fair=args.lambda_fair,
        logging_steps=args.logging_steps,
    )
    print("Negotiation Environment created")
    train_dataset = negotiation_env.create_dataset(size=args.train_size)
    eval_dataset = negotiation_env.create_eval_dataset()
    reward_functions = negotiation_env.get_reward_functions()

    # ---- Training Config ----
    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/{args.run_name}",
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        num_iterations=1,
        max_completion_length=200,
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
        loss_type="grpo",
        temperature=args.temperature,
    )

    # ---- BitsAndBytes Config ----
    if args.quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        bnb_config = None

    print("Training Args:\n", training_args)

    # ---- Load Model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
   
    tokenizer.padding_side = "left"
   
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # ---- PEFT Config ----
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # ---- Create Trainer ----
    trainer = MultiTurnGRPOTrainer(
        model=model,
        reward_funcs=reward_functions, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        # Multi-turn specific (not in Luca's original)
        max_negotiation_rounds=args.max_rounds,
        max_tokens_per_turn=args.max_tokens_per_turn,
        opponent_model=args.opponent_model,
        # LA-GRPO
        turn_level_sampling=args.turn_level_sampling,
        turn_sampling_p=args.turn_sampling_p,
        debug_prints=args.debug,
    )

    # ---- Train ----
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Save ----
    trainer.save_model(os.path.join(args.output_dir, args.run_name, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.run_name, "final"))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Same as Luca's args
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--run-name", type=str, default="grpo_multiturn_1")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    parser.add_argument("--quantized", action="store_true", default=False)

    # Extended args for single-GPU multi-turn
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-tokens-per-turn", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-self", type=float, default=1.0)
    parser.add_argument("--lambda-welfare", type=float, default=0.0)
    parser.add_argument("--lambda-fair", type=float, default=0.0)
    parser.add_argument("--opponent-model", type=str, default=None)
    parser.add_argument("--turn-level-sampling", action="store_true", default=False,
                        help="Enable LA-GRPO turn-level sampling for local credit assignment")
    parser.add_argument("--turn-sampling-p", type=float, default=0.3,
                        help="Geometric distribution parameter for turn sampling (default: 0.3, mean ~2.3)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    # Logging & Saving
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--use-wandb", action="store_true", default=False)

    # Quick test mode
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print per-sample debug info: conversation with mask status, "
                             "decoded masked text, per-sample rewards & advantages")

    args = parser.parse_args()

    if args.save_steps is None:
        args.save_steps = args.eval_steps

    if args.test:
        args.train_size = 100
        args.eval_size = 10
        args.num_generations = 2
        args.max_rounds = 2
        args.save_steps = 999
        args.eval_steps = 999
        args.logging_steps = 1
        args.quantized = True
        print("*** TEST MODE ***\n")

    main(args)