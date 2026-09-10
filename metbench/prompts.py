"""Original evaluation prompts for text/image inputs."""

import random


def build_context(
    moves,
    states,
    image_actions,
    context_type,
    mix_ratio,
    domain,
    ground_truth=None,
    rng=None,
    combine_images=None,
    image_message_layout=None,
):
    """
    Build the messages for the LLM call, domain-dependent.
    - If domain=='chess', we talk about 'chess moves' and final FEN.
    - If domain=='shell', we talk about 'shell swaps' and final 'shell arrangement' or ball position.

    Args:
        moves: List of move strings
        states: List of states
        image_actions: List of base64-encoded images
        context_type: Prompt style (zero-shot or chain-of-thought); Chess also
            supports fen-trace.
        mix_ratio: Ratio of images to use (0.0 = text only, 1.0 = all images)
        domain: Domain (chess or shell)
        ground_truth: Optional ground truth for few-shot examples
        rng: Random number generator for sampling
        combine_images: Compatibility argument; unused.
        image_message_layout: Compatibility argument; unused.
    """
    if domain == "chess":
        domain_instructions = "You are a helpful assistant that tracks chess moves in a game and produces the final FEN.\n"
        if context_type == "fen-trace":
            final_question = "Track the game one move at a time. After each move, write the board placement FEN using exactly this format: BOARD_AFTER_MOVE_1: [board placement FEN], BOARD_AFTER_MOVE_2: [board placement FEN], and so on. A board placement FEN has 8 ranks separated by '/', with digits for empty squares. After all moves, output the final full FEN as FINAL ANSWER: [FEN HERE]."
        elif context_type == "chain-of-thought":
            final_question = "What is the final FEN? Think step by step then output the final FEN as FINAL ANSWER: [FEN HERE]."
        else:
            final_question = "Now what is the final FEN? Only output the FEN and nothing else. If there are no moves you must still just predict the FEN. Start the next message with FINAL ANSWER:"
    elif domain == "shell":
        domain_instructions = "The shell game is a classic game where a ball is hidden under one of three shells. You are a helpful assistant that tracks the position of the ball during the swaps and determines the final position of the ball. We label the shells 1, 2, 3. The ball starts under one of the numbered shells which we call the initial state, and each move is shell swap of shell x and y written 'x swap y'.\n"
        if context_type == "chain-of-thought":
            final_question = "What is the final position of the ball? Think step by step then output the final ball location as FINAL ANSWER: [1, 2, or 3]."
        else:
            final_question = "Now what is the final position of the ball? Only output the final ball location as a single number 1, 2, or 3. If there are no moves you must still just predict the location. Start the next message with FINAL ANSWER:"
    elif domain.startswith("minecraft"):
        raise ValueError(
            "Minecraft domain must be handled via build_minecraft_context(). Do not call build_context() directly for minecraft."
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")
    initial_state = states[0]
    return _build_interleaved_context(
        moves=moves,
        image_actions=image_actions,
        domain_instructions=domain_instructions,
        final_question=final_question,
        initial_state=initial_state,
        domain=domain,
        mix_ratio=mix_ratio,
        rng=rng,
        ground_truth=ground_truth,
    )


def _build_interleaved_context(
    moves,
    image_actions,
    domain_instructions,
    final_question,
    initial_state,
    domain,
    mix_ratio,
    rng,
    ground_truth,
):
    """Build context with images interleaved based on mix_ratio."""
    content_parts = []
    current_text_block = f"{domain_instructions}\n{final_question}\nThe initial state is: {initial_state}\nHere are the moves played:\n"
    rng = rng or random
    for i, move in enumerate(moves):
        use_image = rng.random() < mix_ratio
        if use_image and i < len(image_actions):
            if current_text_block:
                content_parts.append({"type": "text", "text": current_text_block})
                current_text_block = ""
            img_b64 = image_actions[i]
            if domain == "chess":
                caption = "The move is from the green square to the red square."
            else:
                caption = (
                    "The shells with the numbers highlighted green are being swapped."
                )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
            current_text_block = caption + "\n"
        else:
            current_text_block += f"{move}\n"
    current_text_block += f"\n{final_question}"
    if current_text_block:
        content_parts.append({"type": "text", "text": current_text_block})
    has_images = any((p["type"] == "image_url" for p in content_parts))
    if not has_images and len(content_parts) == 1:
        final_user_content = content_parts[0]["text"]
    else:
        final_user_content = content_parts
    messages = [{"role": "user", "content": final_user_content}]
    if ground_truth is not None:
        messages.append(
            {"role": "assistant", "content": f"FINAL ANSWER: {ground_truth}"}
        )
    return messages


def build_minecraft_context(
    entry_data, context_type="zero-shot", ground_truth=None, mode="visual"
):
    """
    Build messages for the Minecraft forward-prediction task.

    The evaluator uses "visual" for screenshots with a text action and
    "text-only" for JSON states with a text action.

    Args:
        entry_data: Task inputs. Both modes require ``action`` and ``choices``;
            the length of ``choices`` determines the number of candidates.
            Visual input uses ``input_image`` and base64 PNGs in ``choices``.
            Text input uses ``input_state`` and ``choice_states``; the values
            in ``choices`` are placeholders in this mode.
        context_type: Prompt style (zero-shot or chain-of-thought).
        ground_truth: Optional answer appended as an assistant turn for
            few-shot examples; omitted by the benchmark evaluator.
        mode: Input representation. The CLI uses "visual" or "text-only".
    """
    mode = _normalize_minecraft_mode(mode)
    action = _minecraft_text_action(entry_data, mode)
    num_choices = len(entry_data["choices"])
    if mode in {"text-only", "image2text"}:
        return _build_minecraft_text_context(
            entry_data, action, num_choices, context_type, ground_truth
        )
    input_b64 = entry_data["input_image"]
    choices_b64 = entry_data["choices"]
    action_image_b64 = entry_data.get("action_image")
    use_action_image = mode == "all-image"
    if use_action_image and (not action_image_b64):
        raise ValueError(
            "Minecraft mode 'all-image' requires entry_data['action_image']."
        )
    domain_instructions = "You are evaluating a Minecraft gameplay trajectory. You will see a first-person screenshot from the game and an action that was performed. Your task is to identify which of the candidate images shows the correct next frame after the action was taken."
    if context_type == "chain-of-thought":
        action_ref = (
            "the action shown in the action image"
            if use_action_image
            else f'the action "{action}"'
        )
        final_question = f"Which image (1-{num_choices}) shows the correct next frame after {action_ref}? Think step by step about how the scene should change given the action, then output your answer as FINAL ANSWER: [1-{num_choices}]."
    else:
        action_ref = (
            "the action shown in the action image"
            if use_action_image
            else f'the action "{action}"'
        )
        final_question = f"Which image (1-{num_choices}) shows the correct next frame after {action_ref}? Output only a single number 1-{num_choices}. Start the next message with FINAL ANSWER:"
    if use_action_image:
        content_parts = [
            {"type": "text", "text": f"{domain_instructions}\n\nAction performed:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{action_image_b64}"},
            },
            {"type": "text", "text": "\nInput frame:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{input_b64}"},
            },
            {"type": "text", "text": "\nCandidate next frames:"},
        ]
    else:
        content_parts = [
            {
                "type": "text",
                "text": f"{domain_instructions}\n\nAction performed: {action}\n\nInput frame:",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{input_b64}"},
            },
            {"type": "text", "text": "\nCandidate next frames:"},
        ]
    for i, choice_b64 in enumerate(choices_b64, start=1):
        content_parts.append({"type": "text", "text": f"\nChoice {i}:"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{choice_b64}"},
            }
        )
    content_parts.append({"type": "text", "text": f"\n{final_question}"})
    messages = [{"role": "user", "content": content_parts}]
    if ground_truth is not None:
        messages.append(
            {"role": "assistant", "content": f"FINAL ANSWER: {ground_truth}"}
        )
    return messages


def _normalize_minecraft_mode(mode):
    """Return the canonical Minecraft mode name or raise on unsupported modes."""
    aliases = {
        "visual": "visual",
        "image-only": "visual",
        "text-only": "text-only",
        "all-image": "all-image",
        "text-image-action": "text-image-action",
        "image2text": "image2text",
        "mc-text-state-text-action": "text-only",
        "mc-image-state-text-action": "visual",
        "mc-image-state-image-action": "all-image",
        "mc-text-state-image-action": "text-image-action",
        "mc-image2text-action": "image2text",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported Minecraft mode '{mode}'. Expected one of: {valid}"
        ) from exc


def _minecraft_text_action(entry_data, mode):
    """Return the text action for modes that use one."""
    if mode in {"all-image", "text-image-action"}:
        return entry_data.get("action", "")
    if mode == "image2text":
        predicted = entry_data.get("image2text_actions")
        if isinstance(predicted, list):
            predicted = predicted[0] if predicted else None
        if not predicted:
            raise ValueError(
                "Minecraft mode 'image2text' requires a predicted action in 'image2text_actions'."
            )
        return predicted
    return entry_data["action"]


def _build_minecraft_text_context(
    entry_data, action, num_choices, context_type, ground_truth
):
    """Build text-only messages for Minecraft forward prediction."""
    import json as _json

    input_state = entry_data.get("input_state", "")
    choice_states = entry_data.get("choice_states", [])
    if isinstance(choice_states, str):
        choice_states = _json.loads(choice_states)
    domain_instructions = "You are evaluating a Minecraft gameplay trajectory. You will see a JSON game-state snapshot and an action that was performed. Your task is to identify which of the candidate game states is the correct next state after the action was taken."
    if context_type == "chain-of-thought":
        final_question = f'Which game state (1-{num_choices}) is the correct next state after the action "{action}"? Think step by step about how the game state should change given the action, then output your answer as FINAL ANSWER: [1-{num_choices}].'
    else:
        final_question = f'Which game state (1-{num_choices}) is the correct next state after the action "{action}"? Output only a single number 1-{num_choices}. Start the next message with FINAL ANSWER:'
    text = f"{domain_instructions}\n\nAction performed: {action}\n\nInput state:\n{input_state}\n\nCandidate next states:"
    for i, state in enumerate(choice_states, start=1):
        text += f"\n\nChoice {i}:\n{state}"
    text += f"\n\n{final_question}"
    messages = [{"role": "user", "content": text}]
    if ground_truth is not None:
        messages.append(
            {"role": "assistant", "content": f"FINAL ANSWER: {ground_truth}"}
        )
    return messages
