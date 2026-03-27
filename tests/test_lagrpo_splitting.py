"""Test LA-GRPO conversation splitting: prefix shared, continuations diverge, mask targets correct turn."""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def mock_play_prefix(prompt_agent, prompt_opponent, agent_starts, num_rounds):
    """Simulate _play_prefix with deterministic responses."""
    agent_history = [{"role": "system", "content": prompt_agent}]
    opponent_history = [{"role": "system", "content": prompt_opponent}]
    conversation = []
    agent_turn_indices = []

    for round_num in range(num_rounds):
        speakers = ["agent", "opponent"] if agent_starts else ["opponent", "agent"]
        for speaker in speakers:
            if speaker == "agent":
                response = f"PREFIX_AGENT_R{round_num}"
                agent_history.append({"role": "assistant", "content": response})
                opponent_history.append({"role": "user", "content": response})
                agent_turn_indices.append(len(conversation))
                conversation.append({"role": "assistant", "content": response})
            else:
                response = f"PREFIX_OPP_R{round_num}"
                opponent_history.append({"role": "assistant", "content": response})
                agent_history.append({"role": "user", "content": response})
                conversation.append({"role": "user", "content": response})

    return {
        "agent_history": agent_history,
        "opponent_history": opponent_history,
        "conversation": conversation,
        "agent_turn_indices": agent_turn_indices,
        "next_round": num_rounds,
    }


def mock_play_negotiation(prompt_agent, prompt_opponent, agent_starts, resume_state, gen_id, max_rounds=5):
    """Simulate _play_negotiation with gen_id-unique responses after prefix."""
    if resume_state is not None:
        agent_history = [dict(m) for m in resume_state["agent_history"]]
        opponent_history = [dict(m) for m in resume_state["opponent_history"]]
        conversation = [dict(m) for m in resume_state["conversation"]]
        agent_turn_indices = list(resume_state["agent_turn_indices"])
        start_round = resume_state["next_round"]
    else:
        agent_history = [{"role": "system", "content": prompt_agent}]
        opponent_history = [{"role": "system", "content": prompt_opponent}]
        conversation = []
        agent_turn_indices = []
        start_round = 0

    for round_num in range(start_round, max_rounds):
        speakers = ["agent", "opponent"] if agent_starts else ["opponent", "agent"]
        for speaker in speakers:
            if speaker == "agent":
                response = f"GEN{gen_id}_AGENT_R{round_num}"
                agent_history.append({"role": "assistant", "content": response})
                opponent_history.append({"role": "user", "content": response})
                agent_turn_indices.append(len(conversation))
                conversation.append({"role": "assistant", "content": response})
            else:
                response = f"GEN{gen_id}_OPP_R{round_num}"
                opponent_history.append({"role": "assistant", "content": response})
                agent_history.append({"role": "user", "content": response})
                conversation.append({"role": "user", "content": response})

    return conversation, agent_turn_indices


def test_splitting():
    max_rounds = 5
    num_generations = 4
    prompt_agent = "You are agent A."
    prompt_opponent = "You are agent B."

    for sampled_h in range(max_rounds):
        print("=" * 70)
        print(f"sampled_h = {sampled_h}")
        print("=" * 70)

        # Step 1: Generate prefix
        prefix_state = None
        mask_from_agent_turn = None
        if sampled_h > 0:
            mask_from_agent_turn = sampled_h
            prefix_state = mock_play_prefix(
                prompt_agent, prompt_opponent, agent_starts=True, num_rounds=sampled_h
            )
            print(f"Prefix conversation ({len(prefix_state['conversation'])} turns):")
            for t in prefix_state["conversation"]:
                print(f"  {t['role']}: {t['content']}")
            print(f"Prefix agent_turn_indices: {prefix_state['agent_turn_indices']}")
            print()

        # Step 2: Generate G continuations
        all_conversations = []
        all_agent_indices = []
        for gen_id in range(num_generations):
            conv, indices = mock_play_negotiation(
                prompt_agent, prompt_opponent,
                agent_starts=True,
                resume_state=prefix_state,
                gen_id=gen_id,
                max_rounds=max_rounds,
            )
            all_conversations.append(conv)
            all_agent_indices.append(indices)

        # Step 3: Verify prefix is shared across all generations
        if sampled_h > 0:
            prefix_len = len(prefix_state["conversation"])
            for gen_id in range(num_generations):
                conv = all_conversations[gen_id]
                for t in range(prefix_len):
                    assert conv[t]["content"] == all_conversations[0][t]["content"], (
                        f"FAIL: gen {gen_id} prefix turn {t} differs!"
                    )
            print(f"[PASS] All {num_generations} generations share the same prefix ({prefix_len} turns)")

        # Step 4: Verify continuations diverge
        if sampled_h < max_rounds:
            diverge_idx = sampled_h * 2  # first agent turn after prefix
            for gen_id in range(1, num_generations):
                assert all_conversations[gen_id][diverge_idx]["content"] != all_conversations[0][diverge_idx]["content"], (
                    f"FAIL: gen {gen_id} should diverge at turn {diverge_idx}!"
                )
            print(f"[PASS] Continuations diverge from conversation index {diverge_idx} (round {sampled_h})")

        # Step 5: Verify agent_turn_indices are correct
        for gen_id in range(num_generations):
            indices = all_agent_indices[gen_id]
            for idx in indices:
                assert all_conversations[gen_id][idx]["role"] == "assistant", (
                    f"FAIL: gen {gen_id} index {idx} is not assistant!"
                )
            assert len(indices) == max_rounds, (
                f"FAIL: gen {gen_id} has {len(indices)} agent turns, expected {max_rounds}"
            )
        print(f"[PASS] All agent_turn_indices point to assistant turns, {max_rounds} each")

        # Step 6: Verify mask targets the correct divergent turn
        print(f"mask_from_agent_turn = {mask_from_agent_turn}")
        for gen_id in range(num_generations):
            indices = all_agent_indices[gen_id]
            if mask_from_agent_turn is not None:
                target_conv_idx = indices[mask_from_agent_turn]
                target_content = all_conversations[gen_id][target_conv_idx]["content"]
                print(f"  Gen {gen_id}: masked turn content = \"{target_content}\"")
                assert f"GEN{gen_id}" in target_content, (
                    f"FAIL: masked turn should be from divergent generation, got: {target_content}"
                )
            else:
                first_agent_content = all_conversations[gen_id][indices[0]]["content"]
                assert f"GEN{gen_id}" in first_agent_content, (
                    f"FAIL: h=0, first agent turn should be unique per gen"
                )

        if mask_from_agent_turn is not None:
            print(f"[PASS] Masked turn (agent turn {mask_from_agent_turn}) is from divergent part")
        else:
            print(f"[PASS] h=0: no prefix, all turns are independently generated")

        # Step 7: Prefix content must not leak into masked turn
        if mask_from_agent_turn is not None:
            for gen_id in range(num_generations):
                indices = all_agent_indices[gen_id]
                target_idx = indices[mask_from_agent_turn]
                target_content = all_conversations[gen_id][target_idx]["content"]
                assert "PREFIX" not in target_content, (
                    f"FAIL: masked turn contains prefix content!"
                )
            print(f"[PASS] Masked turn does not contain prefix content")

        # Step 8: Verify prefix_state is NOT mutated by any generation
        if prefix_state is not None:
            assert len(prefix_state["conversation"]) == sampled_h * 2, (
                f"FAIL: prefix_state was mutated! "
                f"Expected {sampled_h * 2} turns, got {len(prefix_state['conversation'])}"
            )
            assert len(prefix_state["agent_turn_indices"]) == sampled_h, (
                f"FAIL: prefix agent_turn_indices mutated! "
                f"Expected {sampled_h}, got {len(prefix_state['agent_turn_indices'])}"
            )
            for t in prefix_state["conversation"]:
                assert "PREFIX" in t["content"], (
                    f"FAIL: prefix_state content was overwritten: {t['content']}"
                )
            print(f"[PASS] prefix_state was not mutated by any generation")

        # Step 9: Print full conversation for one generation to visualize
        print(f"\n  Full conversation for gen 0:")
        for j, t in enumerate(all_conversations[0]):
            marker = ""
            if j in all_agent_indices[0]:
                agent_num = all_agent_indices[0].index(j)
                if mask_from_agent_turn is not None and agent_num == mask_from_agent_turn:
                    marker = " <<< MASKED FOR LOSS"
                elif mask_from_agent_turn is None:
                    marker = " <<< MASKED FOR LOSS"
            print(f"    [{j}] {t['role']}: {t['content']}{marker}")

        print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_splitting()
