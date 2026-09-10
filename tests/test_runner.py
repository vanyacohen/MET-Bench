import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import evaluate
from metbench.data import select_rows


class RunnerTests(unittest.TestCase):
    def examples(self, domain, limit, actions):
        return select_rows(
            [
                dict(
                    example_id="shell-test-000000",
                    initial_state=1,
                    actions=["1 swap 2"] * 100,
                    states=[1] + [2, 1] * 50,
                    final_state=1,
                )
            ],
            "shell",
            1,
            10,
        )

    def test_resume_reuses_successful_response(self):
        class Backend:
            calls = 0

            def __init__(self, args):
                pass

            def generate(self, messages):
                Backend.calls += 1
                return dict(
                    response="FINAL ANSWER: 1", finish_reason="stop", usage=None
                )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evaluate.load_examples", self.examples),
            patch("metbench.backends.APIBackend", Backend),
        ):
            args = [
                "run",
                "--model",
                "fixture",
                "--domains",
                "shell",
                "--modalities",
                "text",
                "--limit",
                "1",
                "--output",
                tmp,
            ]
            self.assertEqual(evaluate.main(args), 0)
            self.assertEqual(evaluate.main(args + ["--resume"]), 0)
            self.assertEqual(Backend.calls, 1)
            summary = json.loads((Path(tmp) / "summary.json").read_text())["shell/text"]
            self.assertEqual(summary["accuracy"], 1)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                len((Path(tmp) / "shell-text.jsonl").read_text().splitlines()), 1
            )

    def test_api_failure_is_incomplete_and_stops(self):
        class Denied(Exception):
            status_code = 401

        class Backend:
            calls = 0

            def __init__(self, args):
                pass

            def generate(self, messages):
                Backend.calls += 1
                raise Denied("sensitive request content")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evaluate.load_examples", self.examples),
            patch("metbench.backends.APIBackend", Backend),
        ):
            self.assertEqual(
                evaluate.main(
                    [
                        "run",
                        "--model",
                        "fixture",
                        "--domains",
                        "shell",
                        "--limit",
                        "1",
                        "--output",
                        tmp,
                    ]
                ),
                1,
            )
            self.assertEqual(Backend.calls, 1)
            text = (Path(tmp) / "shell-text.jsonl").read_text()
            self.assertNotIn("sensitive", text)
            summary = json.loads((Path(tmp) / "summary.json").read_text())["shell/text"]
            self.assertEqual(summary["completed"], 0)
            self.assertEqual(summary["status"], "incomplete")
