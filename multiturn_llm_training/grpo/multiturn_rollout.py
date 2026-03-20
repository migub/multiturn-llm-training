"""
Multi-turn negotiation rollout function for TRL's GRPOTrainer with vLLM colocate mode.

Uses vLLM for batched agent generation and model.generate() with LoRA disabled
for opponent turns (hybrid approach — no separate GPU needed).

Returns prompt_ids, completion_ids, env_mask (assistant mask), and structured
conversation data for the reward function.
"""

import logging
import time
from functools import partial

import torch

logger = logging.getLogger(__name__)


def create_multiturn_rollout(
    max_negotiation_rounds: int = 5,
    max_tokens_per_turn: int = 200,
    opponent_model: str | None = None,
):
    """Factory that returns a rollout function with the correct TRL signature."""

    def multiturn_negotiation_rollout(prompts, trainer):
        """
        RolloutFunc signature: (prompts: list[str], trainer: GRPOTrainer) -> dict

        prompts: list of prompt strings (already duplicated num_generations times by TRL)
        trainer: the GRPOTrainer instance (has .model, .vllm_generation, .processing_class, etc.)

        Returns dict with:
          - prompt_ids: list[list[int]]
          - completion_ids: list[list[int]]
          - logprobs: None (importance sampling disabled)
          - env_mask: list[list[int]] — 1=agent token, 0=opponent token
          - conversations: list[list[dict]] — structured dialogue for reward function
        """
        device = trainer.accelerator.device
        tokenizer = trainer.processing_class
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        vllm_llm = trainer.vllm_generation.llm
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        # Import vllm SamplingParams
        from vllm import SamplingParams
        agent_sampling = SamplingParams(
            max_tokens=max_tokens_per_turn,
            temperature=trainer.args.temperature if trainer.args.temperature else 1.0,
            top_p=trainer.args.top_p if trainer.args.top_p else 1.0,
            top_k=trainer.args.top_k if trainer.args.top_k else -1,
        )

        # Build GenerationConfig for opponent (local model.generate())
        from transformers import GenerationConfig
        opponent_gen_config = GenerationConfig(
            max_new_tokens=max_tokens_per_turn,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=trainer.args.temperature if trainer.args.temperature else 1.0,
            top_p=trainer.args.top_p if trainer.args.top_p else 1.0,
            top_k=trainer.args.top_k if trainer.args.top_k else 50,
        )

        batch_size = len(prompts)
        logger.info(f"Rollout: generating {batch_size} dialogues, {max_negotiation_rounds} rounds")
        gen_start = time.time()

        # Look up prompt_2 for opponent from the mapping stored on the trainer.
        # This mapping is built from the dataset before training starts.
        prompt_2_map = getattr(trainer, '_prompt_2_map', {})

        # Initialize dialogue histories
        agent_histories = []
        opponent_histories = []
        conversations = [[] for _ in range(batch_size)]
        agent_turn_indices = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            agent_prompt = prompts[i]
            agent_histories.append([{"role": "system", "content": agent_prompt}])
            # Look up opponent prompt; fall back to agent prompt if not found
            opp_prompt = prompt_2_map.get(agent_prompt, agent_prompt)
            opponent_histories.append([{"role": "system", "content": opp_prompt}])

        # --- Turn-by-turn generation ---
        for round_num in range(max_negotiation_rounds):
            # === AGENT TURNS (batched via vLLM) ===
            agent_chat_messages = []
            for i in range(batch_size):
                agent_chat_messages.append(agent_histories[i])

            agent_outputs = vllm_llm.chat(
                messages=agent_chat_messages,
                sampling_params=agent_sampling,
                use_tqdm=False,
            )

            for i in range(batch_size):
                response = agent_outputs[i].outputs[0].text.strip()
                agent_histories[i].append({"role": "assistant", "content": response})
                opponent_histories[i].append({"role": "user", "content": response})
                agent_turn_indices[i].append(len(conversations[i]))
                conversations[i].append({"role": "assistant", "content": response})

            # === OPPONENT TURNS ===
            if opponent_model is not None:
                # OpenAI API opponent
                for i in range(batch_size):
                    response = _openai_response(
                        opponent_histories[i], opponent_model, max_tokens_per_turn,
                        trainer.args.temperature if trainer.args.temperature else 1.0,
                    )
                    opponent_histories[i].append({"role": "assistant", "content": response})
                    agent_histories[i].append({"role": "user", "content": response})
                    conversations[i].append({"role": "user", "content": response})
            else:
                # Local opponent: model.generate() with LoRA disabled
                # Disable gradient checkpointing to allow KV caching during generation
                gc_was_enabled = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
                if not gc_was_enabled:
                    gc_was_enabled = getattr(getattr(unwrapped_model, 'base_model', None), 'is_gradient_checkpointing', False)
                if gc_was_enabled:
                    unwrapped_model.base_model.model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True

                unwrapped_model.disable_adapter_layers()
                for i in range(batch_size):
                    response = _local_generate(
                        unwrapped_model, tokenizer, opponent_histories[i],
                        opponent_gen_config, device,
                    )
                    opponent_histories[i].append({"role": "assistant", "content": response})
                    agent_histories[i].append({"role": "user", "content": response})
                    conversations[i].append({"role": "user", "content": response})
                unwrapped_model.enable_adapter_layers()

                # Re-enable gradient checkpointing for training
                if gc_was_enabled:
                    unwrapped_model.base_model.model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False

        gen_time = time.time() - gen_start
        logger.info(f"Rollout: generation done in {gen_time:.1f}s")
        print(f"Rollout: {batch_size} dialogues generated in {gen_time:.1f}s")

        # --- Tokenize and build env_mask ---
        all_prompt_ids = []
        all_completion_ids = []
        all_env_mask = []

        for i in range(batch_size):
            prompt_ids, completion_ids, env_mask = _tokenize_and_mask(
                tokenizer, prompts[i], conversations[i], agent_turn_indices[i]
            )
            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            all_env_mask.append(env_mask)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": None,
            "env_mask": all_env_mask,
            # Extra fields forwarded to reward function via reward_kwargs
            "conversations": conversations,
        }

    return multiturn_negotiation_rollout


def _local_generate(model, tokenizer, messages, gen_config, device):
    """Generate a single turn response using the local model."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1800
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _openai_response(messages, model_name, max_tokens, temperature):
    """Generate opponent response via OpenAI API."""
    import openai

    client = openai.OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI error: {e}")
        return "I accept your offer."


def _tokenize_and_mask(tokenizer, system_prompt, conversation, agent_turn_indices):
    """
    Tokenize a conversation and build env_mask (1=agent, 0=opponent).

    Returns:
        prompt_ids: list[int] — tokenized system prompt
        completion_ids: list[int] — tokenized dialogue (after system prompt)
        env_mask: list[int] — same length as completion_ids, 1=agent token, 0=opponent
    """
    # Tokenize system prompt (= the "prompt" portion)
    system_messages = [{"role": "system", "content": system_prompt}]
    system_text = tokenizer.apply_chat_template(
        system_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(system_text, truncation=True, max_length=1800)["input_ids"]

    # Tokenize full conversation
    full_messages = [{"role": "system", "content": system_prompt}] + conversation
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, truncation=True, max_length=2048)["input_ids"]

    # completion_ids = everything after the prompt
    completion_ids = full_ids[len(prompt_ids):]

    # Build env_mask by incrementally tokenizing each turn
    env_mask = [0] * len(completion_ids)

    current_messages = [{"role": "system", "content": system_prompt}]
    for i, msg in enumerate(conversation):
        prev_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        current_messages.append(msg)
        curr_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=False
        )

        prev_len = len(tokenizer(prev_text, truncation=True, max_length=2048)["input_ids"])
        curr_len = len(tokenizer(curr_text, truncation=True, max_length=2048)["input_ids"])

        if i in agent_turn_indices:
            # Mark agent tokens in completion_ids
            start = prev_len - len(prompt_ids)
            end = curr_len - len(prompt_ids)
            for j in range(max(0, start), min(end, len(completion_ids))):
                env_mask[j] = 1

    # Fallback: if no agent tokens detected, mark everything as agent
    if sum(env_mask) == 0:
        logger.warning("No agent tokens detected in env_mask — falling back to full mask")
        env_mask = [1] * len(completion_ids)

    return prompt_ids, completion_ids, env_mask
