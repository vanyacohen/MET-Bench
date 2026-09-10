import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import evaluate
from metbench.scoring import (
    aggregate_scores,
    parse_minecraft_choice,
    parse_shell_arrangement,
)


class AnswerParsingTests(unittest.TestCase):
    def test_malformed_and_ambiguous_answers_are_rejected(self):
        for parse in (parse_shell_arrangement, parse_minecraft_choice):
            for response in (
                "FINAL ANSWER: 12",
                "FINAL ANSWER: 1 or 2",
                "I considered shell 2 but cannot determine the answer.",
                "FINAL ANSWER: 0",
                "FINAL ANSWER: 2.5",
            ):
                with self.subTest(parser=parse.__name__, response=response):
                    self.assertIsNone(parse(response))

    def test_formatted_final_answers_and_standalone_choices(self):
        for parse in (parse_shell_arrangement, parse_minecraft_choice):
            for response in (
                "2",
                "FINAL ANSWER: **2**",
                r"FINAL ANSWER: \boxed{2}",
                "FINAL ANSWER: [2]",
                "FINAL ANSWER: Choice 2.",
                "Reasoning mentions 1 and 3.\nFINAL ANSWER: 2",
            ):
                with self.subTest(parser=parse.__name__, response=response):
                    self.assertEqual(parse(response), 2)
        self.assertIsNone(parse_shell_arrangement("4"))
        self.assertEqual(parse_minecraft_choice("4"), 4)

    def test_wilson_interval_uses_actual_bounds(self):
        result = aggregate_scores([dict(invalid_pred=0, successes=1, trials=1)])
        self.assertAlmostEqual(result["accuracy_ci_lower"], 0.2065493143772374)
        self.assertAlmostEqual(result["accuracy_ci_upper"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        empty = aggregate_scores([])
        self.assertEqual(
            (empty["accuracy_ci_lower"], empty["accuracy_ci_upper"]), (0.0, 0.0)
        )


class RecoveryTests(unittest.TestCase):
    @staticmethod
    def examples(domain, limit, actions):
        return [
            dict(
                example_id=f"shell-{i}",
                source_row=i,
                fingerprint=f"input-{i}",
                initial_state=1,
                actions=["1 swap 2"] * actions,
                target=1,
            )
            for i in range(limit)
        ]

    def test_resume_recovers_partial_final_response(self):
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
                "2",
                "--output",
                tmp,
            ]
            self.assertEqual(evaluate.main(args), 0)
            path = Path(tmp) / "shell-text.jsonl"
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(lines[0] + lines[1][:20])
            self.assertEqual(evaluate.main(args + ["--resume"]), 0)
            self.assertEqual(Backend.calls, 3)
            self.assertEqual(len(evaluate.read_predictions(path)), 2)

    def test_valid_last_record_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            path.write_text('{"example_id":"one"}')
            self.assertEqual(
                evaluate.read_predictions(path, repair_trailing=True),
                [{"example_id": "one"}],
            )
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_middle_corruption_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            original = b'{broken}\n{"example_id":"two"}\n'
            path.write_bytes(original)
            with self.assertRaises(json.JSONDecodeError):
                evaluate.read_predictions(path, repair_trailing=True)
            self.assertEqual(path.read_bytes(), original)

    def test_fatal_error_preserves_another_completed_request(self):
        class Denied(Exception):
            status_code = 401

        class Backend:
            calls = 0
            lock = threading.Lock()
            completed = threading.Event()

            def __init__(self, args):
                pass

            def generate(self, messages):
                with self.lock:
                    index = Backend.calls
                    Backend.calls += 1
                if index == 0:
                    if not self.completed.wait(5):
                        raise RuntimeError("Second worker did not finish")
                    raise Denied()
                self.completed.set()
                return dict(
                    response="FINAL ANSWER: 1", finish_reason="stop", usage=None
                )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("evaluate.load_examples", self.examples),
            patch("metbench.backends.APIBackend", Backend),
        ):
            result = evaluate.main(
                [
                    "run",
                    "--model",
                    "fixture",
                    "--domains",
                    "shell",
                    "--modalities",
                    "text",
                    "--limit",
                    "2",
                    "--workers",
                    "2",
                    "--output",
                    tmp,
                ]
            )
            rows = evaluate.read_predictions(Path(tmp) / "shell-text.jsonl")
            self.assertEqual(result, 1)
            self.assertEqual(Backend.calls, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(not row.get("error") for row in rows), 1)
