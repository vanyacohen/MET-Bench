"""Deterministic evaluation selection from the public test splits."""

import base64
import hashlib
import io
import json
from datasets import Image, List, load_dataset
from PIL import Image as PILImage

RELEASES = {
    "chess": ("vanyacohen/MET-Bench-Chess", "904aa58057bf84e0521424f8578f1cae6e6a8ac6"),
    "shell": ("vanyacohen/MET-Bench-Shell", "69dc45b9a0bbf57c3104fca0fc647669bdbf744a"),
    "minecraft": (
        "vanyacohen/MET-Bench-Minecraft",
        "26a94db00630b8ba25f2b23e0db21feffe1f602a",
    ),
}


def fingerprint(value):
    """Hash a JSON-serializable task input using a stable representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_rows(rows, domain, limit=500, actions=10):
    """Keep the first occurrence of each task input in source order."""
    seen = set()
    selected = []
    for index, row in enumerate(rows):
        if domain == "minecraft":
            key = fingerprint(
                [
                    json.loads(row["initial_state"]),
                    row["action"],
                    [json.loads(s) for s in row["candidate_states"]],
                ]
            )
            item = dict(row, target=row["correct_choice"])
        else:
            if len(row["actions"]) < actions or len(row["states"]) <= actions:
                raise ValueError(f"Short trajectory at test row {index}")
            prefix = row["actions"][:actions]
            key = fingerprint([row["initial_state"], prefix])
            item = dict(
                row,
                actions=prefix,
                states=row["states"][: actions + 1],
                target=row["states"][actions],
            )
            item.pop("final_state", None)
            if "image_actions" in row:
                item["image_actions"] = row["image_actions"][:actions]
        if key in seen:
            continue
        seen.add(key)
        item.update(source_row=index, fingerprint=key)
        selected.append(item)
        if len(selected) == limit:
            return selected
    raise ValueError(
        f"Requested {limit} unique {domain} examples; only found {len(selected)}"
    )


def load_examples(domain, limit=500, actions=10):
    """Stream a pinned test split and select unique tasks before decoding images."""
    repo, revision = RELEASES[domain]
    dataset = load_dataset(
        repo, "full", split="test", revision=revision, streaming=True
    )
    # Decode only images actually sent to a model, after truncating trajectories.
    columns = (
        {"image_actions": List(Image(decode=False))}
        if domain != "minecraft"
        else {
            "image_initial_state": Image(decode=False),
            "image_action": Image(decode=False),
            "image_candidate_states": List(Image(decode=False)),
        }
    )
    for name, feature in columns.items():
        dataset = dataset.cast_column(name, feature)
    return select_rows(dataset, domain, limit, actions)


def image_base64(image):
    """Encode dataset image bytes or a loaded image for an API message."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return base64.b64encode(image["bytes"]).decode()
        with PILImage.open(image["path"]) as opened:
            image = opened.copy()
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def messages_for(example, domain, modality, prompt):
    """Build text or image messages from the task inputs, excluding target states."""
    from .prompts import build_context, build_minecraft_context

    if domain == "minecraft":
        entry = dict(
            action=example["action"],
            input_state=example["initial_state"],
            choice_states=example["candidate_states"],
            choices=[None] * 4,
        )
        if modality == "image":
            entry["input_image"] = image_base64(example["image_initial_state"])
            entry["choices"] = [
                image_base64(im) for im in example["image_candidate_states"]
            ]
        return build_minecraft_context(
            entry,
            context_type=prompt,
            mode="text-only" if modality == "text" else "visual",
        )
    return build_context(
        example["actions"],
        [example["initial_state"]],
        (
            [image_base64(im) for im in example["image_actions"]]
            if modality == "image"
            else []
        ),
        prompt,
        1.0 if modality == "image" else 0.0,
        domain,
    )
