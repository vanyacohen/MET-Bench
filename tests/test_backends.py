import unittest
from types import SimpleNamespace
from unittest.mock import patch
from metbench.data import messages_for


class BackendTests(unittest.TestCase):
    def test_api_request_and_response(self):
        import httpx, json
        from openai import OpenAI
        from metbench.backends import APIBackend

        args = SimpleNamespace(
            api_key_env="TEST_EVAL_API_KEY",
            base_url="https://example.test/v1",
            timeout=10,
            model="test-model",
            token_parameter="max_completion_tokens",
            max_tokens=32,
            temperature="0",
            reasoning_effort=None,
        )

        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["model"], "test-model")
            self.assertEqual(body["max_completion_tokens"], 32)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["messages"][0]["content"], "Test prompt")
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "FINAL ANSWER: 2",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        with patch.dict("os.environ", {"TEST_EVAL_API_KEY": "test-only"}):
            backend = APIBackend(args)
        backend.client = OpenAI(
            api_key="test-only",
            base_url=args.base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        result = backend.generate([{"role": "user", "content": "Test prompt"}])
        self.assertEqual(result["response"], "FINAL ANSWER: 2")

    def test_image_prompt_counts(self):
        from PIL import Image

        im = Image.new("RGB", (4, 4))
        shell = dict(
            initial_state=1, actions=["1 swap 2"] * 10, image_actions=[im] * 10
        )
        messages = messages_for(shell, "shell", "image", "zero-shot")
        self.assertEqual(
            sum(p["type"] == "image_url" for p in messages[0]["content"]), 10
        )
        text = " ".join(
            p["text"] for p in messages[0]["content"] if p["type"] == "text"
        )
        self.assertNotIn("1 swap 2", text)
        mc = dict(
            initial_state="{}",
            action="Walk",
            candidate_states=["{}"] * 4,
            image_initial_state=im,
            image_candidate_states=[im] * 4,
        )
        messages = messages_for(mc, "minecraft", "image", "chain-of-thought")
        self.assertEqual(
            sum(p["type"] == "image_url" for p in messages[0]["content"]), 5
        )
