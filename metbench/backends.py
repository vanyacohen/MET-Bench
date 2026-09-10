"""OpenAI-compatible Chat Completions and local Transformers inference."""

import base64
import io
import json
import os
from PIL import Image


class APIBackend:
    """Generate responses through an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, args):
        from openai import OpenAI

        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError(f"Set {args.api_key_env} before using the API backend")
        self.client = OpenAI(
            api_key=key, base_url=args.base_url, timeout=args.timeout, max_retries=3
        )
        self.args = args

    def generate(self, messages):
        """Return response text, completion status, and token usage from the API."""
        args = self.args
        options = dict(model=args.model, messages=messages)
        options[args.token_parameter] = args.max_tokens
        if args.temperature != "omit":
            options["temperature"] = float(args.temperature)
        if args.reasoning_effort:
            options["reasoning_effort"] = args.reasoning_effort
        response = self.client.chat.completions.create(**options)
        choice = response.choices[0]
        return dict(
            response=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=response.usage.model_dump() if response.usage else None,
        )


class HFBackend:
    """Run a local text or vision-language model with Transformers."""

    def __init__(self, args):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        self.args = args
        self.torch = torch
        options = dict(
            device_map=args.device_map, dtype="auto", revision=args.model_revision
        )
        if args.hf_task == "text":
            self.processor = AutoTokenizer.from_pretrained(
                args.model, revision=args.model_revision
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, **options
            ).eval()
        else:
            self.processor = AutoProcessor.from_pretrained(
                args.model,
                revision=args.model_revision,
                **json.loads(args.processor_kwargs),
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model, **options
            ).eval()

    def generate(self, messages):
        """Convert API-style messages to local model inputs and decode the response."""
        converted = []
        for message in messages:
            content = message["content"]
            if self.args.hf_task == "text":
                if not isinstance(content, str):
                    raise ValueError("Use --hf-task vlm for image inputs")
            else:
                parts = (
                    [{"type": "text", "text": content}]
                    if isinstance(content, str)
                    else content
                )
                content = []
                for part in parts:
                    if part["type"] == "text":
                        content.append(part)
                    else:
                        raw = base64.b64decode(
                            part["image_url"]["url"].split(",", 1)[1]
                        )
                        with Image.open(io.BytesIO(raw)) as image:
                            content.append(
                                {"type": "image", "image": image.convert("RGB")}
                            )
            converted.append({"role": message["role"], "content": content})
        inputs = self.processor.apply_chat_template(
            converted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if (
            context_limit
            and inputs["input_ids"].shape[-1] + self.args.max_tokens > context_limit
        ):
            raise ValueError(
                f"Prompt and output budget exceed model context limit ({context_limit}); choose a larger-context model or explicit processor settings"
            )
        options = dict(max_new_tokens=self.args.max_tokens, do_sample=False)
        if self.args.temperature != "omit" and float(self.args.temperature) > 0:
            options.update(do_sample=True, temperature=float(self.args.temperature))
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **options)
        tokens = outputs[:, inputs["input_ids"].shape[-1] :]
        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return dict(
            response=text,
            finish_reason=(
                "length" if tokens.shape[-1] >= self.args.max_tokens else "stop"
            ),
            usage={
                "prompt_tokens": inputs["input_ids"].shape[-1],
                "completion_tokens": tokens.shape[-1],
            },
        )
