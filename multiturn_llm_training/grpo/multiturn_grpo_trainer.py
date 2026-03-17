"""
Multi-Turn GRPO Trainer for Negotiation.

Subclasses the standard TRL GRPOTrainer — NO TRL fork required.
Only overrides _generate() to play multi-turn negotiation dialogues.
Uses TRL's built-in tool_mask mechanism (1 = train, 0 = ignore) as assistant_mask.

Everything else (loss, advantages, KL penalty, logging, checkpointing) comes from TRL.
"""

import copy
import torch
import logging
from typing import Any, Optional
from collections.abc import Callable

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset, IterableDataset
from trl import GRPOTrainer, GRPOConfig

try:
    from peft import PeftConfig
except ImportError:
    PeftConfig = None

logger = logging.getLogger(__name__)


class MultiTurnGRPOTrainer(GRPOTrainer):
    """
    Extends TRL's GRPOTrainer with multi-turn negotiation dialogue generation.
    
    Instead of single-turn prompt→completion, this trainer:
    1. Plays a full negotiation dialogue (Agent vs Opponent, multiple rounds)
    2. Tokenizes the entire dialogue as one sequence
    3. Uses tool_mask (= assistant_mask) to train ONLY on agent tokens
    
    The opponent can be:
    - Local: same model with LoRA disabled (frozen base model)
    - External: OpenAI API (e.g. gpt-4o-mini)
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config: "PeftConfig | None" = None,
        # Multi-turn specific
        max_negotiation_rounds: int = 5,
        max_tokens_per_turn: int = 200,
        opponent_model: str | None = None,
        **kwargs,
    ):
        # Store multi-turn config BEFORE super().__init__ (which may access self)
        self.max_negotiation_rounds = max_negotiation_rounds
        self.max_tokens_per_turn = max_tokens_per_turn
        self.opponent_model = opponent_model
        self._current_inputs = None  # Will hold inputs during generation

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Get the tokenizer (processing_class might be a processor)
        if hasattr(self.processing_class, 'tokenizer'):
            tokenizer = self.processing_class.tokenizer
        else:
            tokenizer = self.processing_class

        # GenerationConfig for per-turn generation
        self.multiturn_generation_config = GenerationConfig(
            max_new_tokens=self.max_tokens_per_turn,
            max_length=None,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=self.temperature if self.temperature else 1.0,
            top_p=self.top_p if self.top_p else 1.0,
            top_k=self.top_k if self.top_k else 50,
        )

        logger.info(
            f"MultiTurnGRPOTrainer initialized: {max_negotiation_rounds} rounds, "
            f"{max_tokens_per_turn} tokens/turn, "
            f"opponent={'OpenAI ' + opponent_model if opponent_model else 'local (LoRA disabled)'}"
        )

    # ----------------------------------------------------------------
    # Override: Allow extra dataset columns (prompt_2, game_config, etc.)
    # ----------------------------------------------------------------
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt", "prompt_2", "game_config", "starting_agent",
                "negotiation_role", "archetype",
                "image", "images",
            ]

    # ----------------------------------------------------------------
    # Override: Store inputs so _generate() can access them
    # ----------------------------------------------------------------
    def _generate_and_score_completions(self, inputs):
        self._current_inputs = inputs
        result = super()._generate_and_score_completions(inputs)
        self._current_inputs = None
        return result

    # ----------------------------------------------------------------
    # Override: Multi-turn dialogue generation instead of single-turn
    # ----------------------------------------------------------------
    def _generate(self, prompts: list):
        """
        Override TRL's single-turn _generate with multi-turn negotiation.
        
        Returns the same tuple format as the parent:
            (prompt_ids, completion_ids, tool_mask, completions,
             total_completion_tokens, logprobs, extra_fields)
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        inputs = self._current_inputs or []

        all_prompt_ids = []
        all_completion_ids = []
        all_assistant_masks = []
        all_completions = []  # For reward function

        for i, prompt in enumerate(prompts):
            inp = inputs[i] if i < len(inputs) else {}

            # Extract agent prompt text
            agent_prompt = prompt
            if isinstance(agent_prompt, list):
                agent_prompt = agent_prompt[-1]["content"] if agent_prompt else ""

            # Extract opponent prompt
            opponent_prompt = inp.get("prompt_2", agent_prompt)
            if isinstance(opponent_prompt, list):
                opponent_prompt = opponent_prompt[-1]["content"] if opponent_prompt else ""

            agent_starts = inp.get("starting_agent", True)

            # Play the negotiation
            conversation, agent_indices = self._play_negotiation(
                prompt_agent=agent_prompt,
                prompt_opponent=opponent_prompt,
                agent_starts=agent_starts,
            )

            # Tokenize and get assistant mask
            p_ids, c_ids, a_mask = self._tokenize_multiturn_conversation(
                agent_prompt, conversation, agent_indices
            )

            all_prompt_ids.append(p_ids)
            all_completion_ids.append(c_ids)
            all_assistant_masks.append(a_mask)
            all_completions.append(conversation)

        # --- Metrics ---
        prompt_lengths = torch.tensor([len(ids) for ids in all_prompt_ids], device=device)
        completion_lengths = torch.tensor([sum(m) for m in all_assistant_masks], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (agg_prompt_lengths.sum() + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Agent token ratio
        total_comp = sum(len(c) for c in all_completion_ids)
        total_agent = sum(sum(m) for m in all_assistant_masks)
        self._metrics[mode]["multiturn/agent_token_ratio"].append(total_agent / max(total_comp, 1))
        self._metrics[mode]["multiturn/avg_dialogue_turns"].append(
            sum(len(c) for c in all_completions) / max(len(all_completions), 1)
        )

        # Return in TRL's expected format
        # tool_mask = assistant_mask → TRL will use it in loss: mask = completion_mask * tool_mask
        return (
            all_prompt_ids,       # list of list[int]
            all_completion_ids,   # list of list[int]
            all_assistant_masks,  # list of list[int] — used as tool_mask!
            all_completions,      # list of conversations — passed to reward functions
            total_completion_tokens,
            None,                 # logprobs (not used without vLLM)
            {},                   # extra_fields
        )

    # ----------------------------------------------------------------
    # Multi-turn helper methods
    # ----------------------------------------------------------------

    @torch.no_grad()
    def _play_negotiation(
        self, prompt_agent: str, prompt_opponent: str, agent_starts: bool = True,
    ) -> tuple[list, list]:
        """
        Play a full multi-turn negotiation dialogue.
        
        Agent: model WITH LoRA adapters (trainable policy).
        Opponent: local model (LoRA disabled) OR OpenAI API.

        Returns:
            conversation: List of message dicts [{role, content}, ...]
            agent_turn_indices: Which indices in conversation are the agent's turns
        """
        agent_history = [{"role": "system", "content": prompt_agent}]
        opponent_history = [{"role": "system", "content": prompt_opponent}]
        conversation = []
        agent_turn_indices = []

        unwrapped = self.accelerator.unwrap_model(self.model)

        for round_num in range(self.max_negotiation_rounds):
            speakers = ["agent", "opponent"] if agent_starts else ["opponent", "agent"]

            for speaker in speakers:
                if speaker == "agent":
                    # Agent: WITH LoRA (trainable policy)
                    unwrapped.enable_adapter_layers()
                    response = self._generate_single_turn_response(agent_history)
                    agent_history.append({"role": "assistant", "content": response})
                    opponent_history.append({"role": "user", "content": response})
                    agent_turn_indices.append(len(conversation))
                    conversation.append({"role": "assistant", "content": response})
                else:
                    if self.opponent_model is not None:
                        # Opponent via OpenAI API
                        response = self._openai_opponent_response(opponent_history)
                    else:
                        # Opponent: local model WITHOUT LoRA (frozen base model)
                        unwrapped.disable_adapter_layers()
                        response = self._generate_single_turn_response(opponent_history)
                        unwrapped.enable_adapter_layers()

                    opponent_history.append({"role": "assistant", "content": response})
                    agent_history.append({"role": "user", "content": response})
                    conversation.append({"role": "user", "content": response})

        # Ensure LoRA is re-enabled
        unwrapped.enable_adapter_layers()
        return conversation, agent_turn_indices

    @torch.no_grad()
    def _generate_single_turn_response(self, messages: list) -> str:
        input_text = self.processing_class.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processing_class(
            input_text, return_tensors="pt", truncation=True, max_length=1800
        ).to(self.accelerator.device)

        # Bypass Unsloth's fast path — use standard HF generate
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.multiturn_generation_config,
                use_cache=False,  # Disables Unsloth's custom KV cache
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()

    def _openai_opponent_response(self, messages: list) -> str:
        """Generate opponent response using OpenAI API."""
        import openai
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.opponent_model,
                messages=messages,
                max_tokens=self.max_tokens_per_turn,
                temperature=self.temperature if self.temperature else 1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI opponent error: {e}. Returning fallback.")
            return "I accept your offer."

    def _tokenize_multiturn_conversation(
        self, system_prompt: str, conversation: list, agent_turn_indices: list
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Tokenize a full multi-turn conversation and create an assistant_mask.

        Returns:
            prompt_ids: Token IDs for the system prompt
            completion_ids: Token IDs for the conversation (all turns)
            assistant_mask: 1 = agent token (train), 0 = opponent token (ignore)
        """
        # System prompt tokens
        system_messages = [{"role": "system", "content": system_prompt}]
        system_text = self.processing_class.apply_chat_template(
            system_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.processing_class(system_text, truncation=True, max_length=1800)["input_ids"]

        # Full conversation tokens
        full_messages = [{"role": "system", "content": system_prompt}] + conversation
        full_text = self.processing_class.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.processing_class(full_text, truncation=True, max_length=2048)["input_ids"]

        # Completion = everything after system prompt
        completion_ids = full_ids[len(prompt_ids):]

        # Build assistant_mask by incrementally tokenizing
        assistant_mask = [0] * len(completion_ids)

        current_messages = [{"role": "system", "content": system_prompt}]
        for i, msg in enumerate(conversation):
            current_messages.append(msg)
            if i in agent_turn_indices:
                text_with = self.processing_class.apply_chat_template(
                    current_messages, tokenize=False, add_generation_prompt=False
                )
                text_without = self.processing_class.apply_chat_template(
                    current_messages[:-1], tokenize=False, add_generation_prompt=True
                )
                tokens_with = self.processing_class(text_with, truncation=True, max_length=2048)["input_ids"]
                tokens_without = self.processing_class(text_without, truncation=True, max_length=2048)["input_ids"]

                start = len(tokens_without) - len(prompt_ids)
                end = len(tokens_with) - len(prompt_ids)

                if start >= 0 and end <= len(completion_ids):
                    for j in range(start, end):
                        assistant_mask[j] = 1

        # Fallback: if no agent tokens found, mark ALL to prevent NaN loss
        if sum(assistant_mask) == 0:
            logger.warning(
                f"No agent tokens detected in assistant_mask! "
                f"Conversation: {len(conversation)} messages, agent_indices={agent_turn_indices}. "
                f"Falling back to full mask."
            )
            assistant_mask = [1] * len(completion_ids)

        return prompt_ids, completion_ids, assistant_mask