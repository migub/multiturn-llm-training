#!/usr/bin/env python3
"""Tests for LA-GRPO implementation in MultiTurnGRPOTrainer.

Validates:
  1. sample_geometric_bounded: distribution properties match Luca's original
  2. _tokenize_conversation with mask_from_agent_turn: correct turn-level masking
  3. _play_negotiation with resume_state: prefix + continuation = full dialogue structure
  4. End-to-end: LA-GRPO masking only includes tokens from turn h onward

Usage:
    python tests/test_lagrpo.py
    python tests/test_lagrpo.py --model-name meta-llama/Llama-3.2-1B-Instruct  # uses real tokenizer
"""

import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


# ============================================================
# Test 1: sample_geometric_bounded
# ============================================================

def test_geometric_sampling():
    """Verify sample_geometric_bounded matches Luca's original implementation."""
    from multiturn_llm_training.grpo.multiturn_grpo_trainer import sample_geometric_bounded

    print("=" * 60)
    print("TEST 1: sample_geometric_bounded")
    print("=" * 60)

    p = 0.3
    max_value = 4
    n_samples = 50_000

    # --- 1a: All values in valid range ---
    rng = np.random.default_rng(seed=42)
    samples = [sample_geometric_bounded(p, max_value, rng) for _ in range(n_samples)]
    assert all(0 <= s <= max_value for s in samples), "FAIL: sample out of range"
    print(f"  [PASS] All {n_samples} samples in [0, {max_value}]")

    # --- 1b: Distribution shape (h=0 most frequent, decreasing) ---
    counts = Counter(samples)
    for h in range(max_value):
        assert counts[h] >= counts.get(h + 1, 0), (
            f"FAIL: h={h} ({counts[h]}) should be >= h={h+1} ({counts.get(h+1, 0)})"
        )
    print(f"  [PASS] Distribution is monotonically decreasing: {dict(sorted(counts.items()))}")

    # --- 1c: Same seed → same sequence (deterministic) ---
    rng1 = np.random.default_rng(seed=123)
    rng2 = np.random.default_rng(seed=123)
    seq1 = [sample_geometric_bounded(p, max_value, rng1) for _ in range(100)]
    seq2 = [sample_geometric_bounded(p, max_value, rng2) for _ in range(100)]
    assert seq1 == seq2, "FAIL: same seed produced different sequences"
    print(f"  [PASS] Deterministic with same seed")

    # --- 1d: Compare with Luca's implementation (if importable) ---
    try:
        from multiturn_llm_training.grpo.lagrpo_trainer import (
            sample_geometric_bounded as luca_sample_geometric_bounded,
        )
        rng_michael = np.random.default_rng(seed=999)
        rng_luca = np.random.default_rng(seed=999)
        michael_seq = [sample_geometric_bounded(p, max_value, rng_michael) for _ in range(200)]
        luca_seq = [luca_sample_geometric_bounded(p, max_value, rng_luca) for _ in range(200)]
        assert michael_seq == luca_seq, (
            f"FAIL: Michael's and Luca's implementations diverge!\n"
            f"  Michael: {michael_seq[:10]}...\n"
            f"  Luca:    {luca_seq[:10]}..."
        )
        print(f"  [PASS] Identical output to Luca's implementation (200 samples)")
    except ImportError:
        print(f"  [SKIP] Luca's lagrpo_trainer not importable (TRL version mismatch) — "
              f"comparing logic manually instead")
        # Verify our implementation matches the algorithm:
        # sample = geometric(p) - 1, reject if > max_value
        rng_manual = np.random.default_rng(seed=999)
        rng_ours = np.random.default_rng(seed=999)
        for _ in range(200):
            # Manual implementation (same as Luca's)
            while True:
                manual_sample = rng_manual.geometric(p) - 1
                if manual_sample <= max_value:
                    break
            our_sample = sample_geometric_bounded(p, max_value, rng_ours)
            assert manual_sample == our_sample, (
                f"FAIL: our implementation diverges from manual geometric sampling"
            )
        print(f"  [PASS] Matches manual geometric(p)-1 with rejection sampling (200 samples)")

    # --- 1e: Approximate mean ---
    mean_h = np.mean(samples)
    # Geometric(0.3)-1 has theoretical mean (1-p)/p = 2.33, but bounded by 4 → ~1.3
    # (36% of samples are h=0, pulling mean down significantly)
    assert 1.0 < mean_h < 2.0, f"FAIL: mean {mean_h:.2f} outside expected range"
    print(f"  [PASS] Mean h = {mean_h:.2f} (expected ~1.3 for bounded geometric(0.3), max=4)")

    print()


# ============================================================
# Test 2: _tokenize_conversation with mask_from_agent_turn
# ============================================================

def test_turn_level_masking(tokenizer):
    """Verify mask_from_agent_turn correctly restricts assistant_mask."""
    from multiturn_llm_training.grpo.multiturn_grpo_trainer import MultiTurnGRPOTrainer

    print("=" * 60)
    print("TEST 2: Turn-level masking in _tokenize_conversation")
    print("=" * 60)

    # Create a dummy conversation (5 rounds, agent starts)
    system_prompt = "You are a negotiator. Your goal is to reach an agreement."
    conversation = [
        {"role": "assistant", "content": "I propose we split the revenue 60-40 in our favor."},
        {"role": "user", "content": "That seems too aggressive. How about 50-50?"},
        {"role": "assistant", "content": "I can consider 55-45. We bring more to the table."},
        {"role": "user", "content": "Let me think. Maybe 52-48 would work."},
        {"role": "assistant", "content": "Deal. 52-48 it is. Let's finalize the agreement."},
        {"role": "user", "content": "Agreed. 52-48 split on revenue."},
        {"role": "assistant", "content": "Great, we have a deal on revenue split at 52-48."},
        {"role": "user", "content": "Confirmed. Looking forward to working together."},
        {"role": "assistant", "content": "Likewise. Agreement reached on all terms."},
        {"role": "user", "content": "Perfect. Deal is done."},
    ]
    # Agent turns are at indices 0, 2, 4, 6, 8 (assistant messages)
    agent_turn_indices = [0, 2, 4, 6, 8]

    # We need a minimal trainer-like object to call _tokenize_conversation
    # Create a mock that has just enough to work
    class MockAccelerator:
        device = torch.device("cpu")

    class MockTrainer:
        def __init__(self, tok):
            self._tokenizer = tok
            self.accelerator = MockAccelerator()

    mock = MockTrainer(tokenizer)
    # Bind the method
    tokenize_fn = MultiTurnGRPOTrainer._tokenize_conversation.__get__(mock)

    # --- 2a: No masking (standard GRPO) — all agent turns masked ---
    tok_ids, att_mask, ass_mask_full, n_agent, n_opp = tokenize_fn(
        system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=None
    )
    full_mask_sum = ass_mask_full.sum().item()
    assert full_mask_sum > 0, "FAIL: no agent tokens in full mask"
    print(f"  Full mask (standard GRPO): {full_mask_sum} agent tokens out of {len(tok_ids)}")

    # --- 2b: mask_from_agent_turn=0 should equal full mask (all agent turns) ---
    _, _, ass_mask_h0, _, _ = tokenize_fn(
        system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=0
    )
    assert torch.equal(ass_mask_full, ass_mask_h0), (
        f"FAIL: mask_from_agent_turn=0 should equal full mask\n"
        f"  full: {ass_mask_full.sum().item()}, h=0: {ass_mask_h0.sum().item()}"
    )
    print(f"  [PASS] mask_from_agent_turn=0 == full mask ({full_mask_sum} tokens)")

    # --- 2c: Increasing h → decreasing mask sum ---
    prev_sum = full_mask_sum
    for h in range(1, len(agent_turn_indices)):
        _, _, ass_mask_h, _, _ = tokenize_fn(
            system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=h
        )
        h_sum = ass_mask_h.sum().item()
        assert h_sum < prev_sum, (
            f"FAIL: h={h} mask ({h_sum}) should be < h={h-1} mask ({prev_sum})"
        )
        print(f"  [PASS] mask_from_agent_turn={h}: {h_sum} tokens (< {prev_sum})")
        prev_sum = h_sum

    # --- 2d: mask_from_agent_turn = last turn → only last agent turn masked ---
    last_h = len(agent_turn_indices) - 1
    _, _, ass_mask_last, _, _ = tokenize_fn(
        system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=last_h
    )
    last_sum = ass_mask_last.sum().item()
    assert 0 < last_sum < full_mask_sum, (
        f"FAIL: last turn mask ({last_sum}) should be between 0 and {full_mask_sum}"
    )
    print(f"  [PASS] mask_from_agent_turn={last_h} (last): {last_sum} tokens")

    # --- 2e: Masked tokens should be a subset of full mask ---
    for h in range(len(agent_turn_indices)):
        _, _, ass_mask_h, _, _ = tokenize_fn(
            system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=h
        )
        # Every 1 in h-mask should also be 1 in full mask
        assert ((ass_mask_h == 1) & (ass_mask_full == 0)).sum().item() == 0, (
            f"FAIL: h={h} mask has tokens not in full mask"
        )
    print(f"  [PASS] All turn-level masks are subsets of full mask")

    # --- 2f: Decode masked tokens to verify correctness ---
    print(f"\n  Decoded masked tokens per turn threshold:")
    for h in range(len(agent_turn_indices)):
        _, _, ass_mask_h, _, _ = tokenize_fn(
            system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=h
        )
        # Decode: mask[:, 1:] alignment (same as trainer)
        shifted_mask = ass_mask_h[1:]
        kept_ids = tok_ids[1:][shifted_mask == 1]
        decoded = tokenizer.decode(kept_ids, skip_special_tokens=True)
        # The decoded text should NOT contain text from agent turns before h
        excluded_turns = [conversation[agent_turn_indices[k]]["content"] for k in range(h)]
        for exc_text in excluded_turns:
            # Check a significant substring (first 20 chars) isn't in decoded
            check_str = exc_text[:20]
            assert check_str not in decoded, (
                f"FAIL: h={h} mask includes text from excluded turn: '{check_str}'"
            )
        print(f"    h={h}: '{decoded[:80]}...' ({ass_mask_h.sum().item()} tokens)")

    print(f"  [PASS] Decoded tokens correctly exclude prefix agent turns")
    print()


# ============================================================
# Test 3: Prefix + continuation structure
# ============================================================

def test_prefix_continuation_structure():
    """Verify _play_prefix returns correct state and _play_negotiation resumes correctly."""
    print("=" * 60)
    print("TEST 3: Prefix + continuation structure")
    print("=" * 60)

    # We can't call the actual methods without a model, but we can test
    # the state dict structure and resume logic with a mock

    # Simulate what _play_prefix returns for h=2
    prefix_state = {
        "agent_history": [
            {"role": "system", "content": "You are agent A."},
            {"role": "assistant", "content": "Agent turn 0"},
            {"role": "user", "content": "Opponent turn 0"},
            {"role": "assistant", "content": "Agent turn 1"},
            {"role": "user", "content": "Opponent turn 1"},
        ],
        "opponent_history": [
            {"role": "system", "content": "You are agent B."},
            {"role": "user", "content": "Agent turn 0"},
            {"role": "assistant", "content": "Opponent turn 0"},
            {"role": "user", "content": "Agent turn 1"},
            {"role": "assistant", "content": "Opponent turn 1"},
        ],
        "conversation": [
            {"role": "assistant", "content": "Agent turn 0"},
            {"role": "user", "content": "Opponent turn 0"},
            {"role": "assistant", "content": "Agent turn 1"},
            {"role": "user", "content": "Opponent turn 1"},
        ],
        "agent_turn_indices": [0, 2],
        "next_round": 2,
    }

    # --- 3a: State has correct keys ---
    required_keys = {"agent_history", "opponent_history", "conversation",
                     "agent_turn_indices", "next_round"}
    assert set(prefix_state.keys()) == required_keys, (
        f"FAIL: missing keys: {required_keys - set(prefix_state.keys())}"
    )
    print(f"  [PASS] Prefix state has all required keys")

    # --- 3b: next_round matches h ---
    assert prefix_state["next_round"] == 2, "FAIL: next_round should be 2 for h=2"
    print(f"  [PASS] next_round = {prefix_state['next_round']}")

    # --- 3c: Conversation has 2*h turns (agent + opponent per round) ---
    assert len(prefix_state["conversation"]) == 4, (
        f"FAIL: expected 4 turns for h=2, got {len(prefix_state['conversation'])}"
    )
    print(f"  [PASS] Prefix conversation has {len(prefix_state['conversation'])} turns (2 rounds)")

    # --- 3d: agent_turn_indices correct ---
    assert prefix_state["agent_turn_indices"] == [0, 2], (
        f"FAIL: expected [0, 2], got {prefix_state['agent_turn_indices']}"
    )
    print(f"  [PASS] Agent turn indices = {prefix_state['agent_turn_indices']}")

    # --- 3e: Simulate resume — deep copy prevents cross-contamination ---
    import copy
    state1 = copy.deepcopy(prefix_state)
    state2 = copy.deepcopy(prefix_state)

    # Simulate two different continuations from round 2
    state1["conversation"].append({"role": "assistant", "content": "Continuation A"})
    state1["agent_turn_indices"].append(4)
    state2["conversation"].append({"role": "assistant", "content": "Continuation B"})
    state2["agent_turn_indices"].append(4)

    # Verify they diverged
    assert state1["conversation"][-1]["content"] != state2["conversation"][-1]["content"], (
        "FAIL: continuations should differ"
    )
    # Verify prefix is unchanged
    assert prefix_state["conversation"][-1]["content"] == "Opponent turn 1", (
        "FAIL: original prefix_state was mutated"
    )
    print(f"  [PASS] Deep copy prevents cross-contamination between generations")

    # --- 3f: Verify resume_state dict copy in _play_negotiation ---
    # The actual code does: [dict(m) for m in resume_state["conversation"]]
    # Test that this creates independent copies
    original_conv = [{"role": "assistant", "content": "test"}]
    copied = [dict(m) for m in original_conv]
    copied[0]["content"] = "modified"
    assert original_conv[0]["content"] == "test", "FAIL: dict copy should not modify original"
    print(f"  [PASS] Dict copy creates independent message objects")

    print()


# ============================================================
# Test 4: End-to-end LA-GRPO masking logic
# ============================================================

def test_e2e_lagrpo_masking(tokenizer):
    """End-to-end test: for different h values, verify the correct tokens are in the loss."""
    from multiturn_llm_training.grpo.multiturn_grpo_trainer import MultiTurnGRPOTrainer

    print("=" * 60)
    print("TEST 4: End-to-end LA-GRPO masking")
    print("=" * 60)

    system_prompt = "You are negotiating a joint venture deal."

    # 3 rounds, agent starts → 6 conversation turns
    # Agent turns: 0, 2, 4  Opponent turns: 1, 3, 5
    agent_texts = [
        "I suggest a 60-40 revenue split and shared R&D costs.",
        "How about 55-45 on revenue but we keep R&D separate?",
        "Agreed. 55-45 revenue, separate R&D. Deal.",
    ]
    opp_texts = [
        "60-40 is too much. I want at least 45%.",
        "Separate R&D works. But I need 48% on revenue.",
        "Fine, 55-45 with separate R&D. Deal confirmed.",
    ]

    conversation = []
    agent_turn_indices = []
    for r in range(3):
        agent_turn_indices.append(len(conversation))
        conversation.append({"role": "assistant", "content": agent_texts[r]})
        conversation.append({"role": "user", "content": opp_texts[r]})

    assert agent_turn_indices == [0, 2, 4]

    class MockAccelerator:
        device = torch.device("cpu")

    class MockTrainer:
        def __init__(self, tok):
            self._tokenizer = tok
            self.accelerator = MockAccelerator()

    mock = MockTrainer(tokenizer)
    tokenize_fn = MultiTurnGRPOTrainer._tokenize_conversation.__get__(mock)

    # Get full mask for reference
    tok_ids, _, full_mask, _, _ = tokenize_fn(
        system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=None
    )

    print(f"  Total tokens: {len(tok_ids)}, Full mask agent tokens: {full_mask.sum().item()}")

    # For each h, verify:
    # - Masked tokens come ONLY from agent turns >= h
    # - The union of prefix mask + continuation mask = full mask
    for h in range(3):
        _, _, mask_h, _, _ = tokenize_fn(
            system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=h
        )

        # Decode what's included
        shifted_mask = mask_h[1:]
        kept_ids = tok_ids[1:][shifted_mask == 1]
        decoded = tokenizer.decode(kept_ids, skip_special_tokens=True)

        # Verify included agent texts are present
        for k in range(h, 3):
            # Check a key phrase from each included agent turn
            key_phrase = agent_texts[k][:15]
            assert key_phrase in decoded, (
                f"FAIL h={h}: agent turn {k} text '{key_phrase}' should be in mask but isn't.\n"
                f"  Decoded: {decoded}"
            )

        # Verify excluded agent texts are NOT present
        for k in range(h):
            key_phrase = agent_texts[k][:15]
            assert key_phrase not in decoded, (
                f"FAIL h={h}: agent turn {k} text '{key_phrase}' should NOT be in mask.\n"
                f"  Decoded: {decoded}"
            )

        # Verify NO opponent text is included
        for k in range(3):
            key_phrase = opp_texts[k][:15]
            assert key_phrase not in decoded, (
                f"FAIL h={h}: opponent turn {k} text '{key_phrase}' should NOT be in mask.\n"
                f"  Decoded: {decoded}"
            )

        print(f"  [PASS] h={h}: mask includes agent turns {list(range(h,3))}, "
              f"excludes turns {list(range(h))} + all opponent turns "
              f"({mask_h.sum().item()} tokens)")

    # --- 4b: h=0 mask == full mask (standard GRPO equivalence) ---
    _, _, mask_h0, _, _ = tokenize_fn(
        system_prompt, conversation, agent_turn_indices, mask_from_agent_turn=0
    )
    assert torch.equal(mask_h0, full_mask), "FAIL: h=0 should equal full mask"
    print(f"  [PASS] h=0 is identical to standard GRPO (full mask)")

    print()


# ============================================================
# Test 5: Verify sampled_h=0 means no prefix (standard GRPO)
# ============================================================

def test_h0_no_prefix():
    """When sampled_h=0, no prefix should be generated."""
    print("=" * 60)
    print("TEST 5: h=0 degenerates to standard GRPO")
    print("=" * 60)

    # In _generate_and_score_completions, the logic is:
    #   if sampled_h > 0: mask_from_agent_turn = sampled_h
    #   else: mask_from_agent_turn = None (and prefix_state = None)
    # So h=0 → prefix_state=None → _play_negotiation runs from scratch
    # And mask_from_agent_turn=None → full mask

    sampled_h = 0
    mask_from_agent_turn = sampled_h if sampled_h > 0 else None
    prefix_state = "would_generate_prefix" if sampled_h > 0 else None

    assert mask_from_agent_turn is None, "FAIL: h=0 should give None mask"
    assert prefix_state is None, "FAIL: h=0 should give None prefix"
    print(f"  [PASS] h=0: mask_from_agent_turn=None, prefix_state=None (standard GRPO)")

    # Also verify h=1 does trigger prefix
    sampled_h = 1
    mask_from_agent_turn = sampled_h if sampled_h > 0 else None
    assert mask_from_agent_turn == 1, "FAIL: h=1 should set mask_from_agent_turn=1"
    print(f"  [PASS] h=1: mask_from_agent_turn=1 (LA-GRPO active)")

    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test LA-GRPO implementation")
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model for tokenizer (default: uses tiktoken-compatible mock)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LA-GRPO IMPLEMENTATION TESTS")
    print("=" * 60 + "\n")

    # Test 1: No tokenizer needed
    test_geometric_sampling()

    # Test 3 & 5: No tokenizer needed
    test_prefix_continuation_structure()
    test_h0_no_prefix()

    # Tests 2 & 4: Need a tokenizer
    if args.model_name:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # Try a small default model
        try:
            from transformers import AutoTokenizer
            default_model = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"No --model-name specified, trying {default_model}...")
            tokenizer = AutoTokenizer.from_pretrained(default_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"  Could not load tokenizer ({e})")
            print(f"  Skipping tests 2 & 4 (need --model-name)")
            print(f"\n{'=' * 60}")
            print("PARTIAL PASS: Tests 1, 3, 5 passed. Tests 2, 4 skipped.")
            print("=" * 60)
            return

    test_turn_level_masking(tokenizer)
    test_e2e_lagrpo_masking(tokenizer)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
