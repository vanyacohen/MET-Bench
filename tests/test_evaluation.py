import copy
import unittest
import chess
from metbench.data import select_rows, messages_for
from metbench.scoring import (
    score_single_prediction_chess,
    score_single_prediction_shell,
    score_single_prediction_minecraft,
)


class SelectionTests(unittest.TestCase):
    def shell(self, initial=1):
        return dict(
            example_id="shell-test-000000",
            initial_state=initial,
            actions=["1 swap 2"] * 100,
            states=[initial] + [2, 1] * 50,
            final_state=1,
            image_actions=list(range(100)),
        )

    def test_prefix_deduplication_and_target(self):
        first = self.shell()
        first["states"][10] = 2
        duplicate = copy.deepcopy(first)
        duplicate["actions"][99] = "2 swap 3"
        distinct = self.shell(3)
        result = select_rows([first, duplicate, distinct], "shell", 2, 10)
        self.assertEqual([x["source_row"] for x in result], [0, 2])
        self.assertEqual(result[0]["target"], 2)
        self.assertEqual(len(result[0]["image_actions"]), 10)
        self.assertEqual(len(result[0]["states"]), 11)
        self.assertNotIn("final_state", result[0])

    def test_insufficient_unique_rows_fails(self):
        with self.assertRaises(ValueError):
            select_rows([self.shell()] * 3, "shell", 2)

    def test_prompt_does_not_include_future_or_gold_states(self):
        row = select_rows([self.shell()], "shell", 1)[0]
        row["target"] = "SECRET_TARGET"
        row["states"] = ["SECRET_STATES"] * 11
        text = messages_for(row, "shell", "text", "zero-shot")[0]["content"]
        self.assertNotIn("SECRET", text)
        self.assertEqual(text.count("1 swap 2"), 10)

    def test_minecraft_uses_choice_not_action_prefix(self):
        row = dict(
            example_id="mc",
            initial_state='{"x":1}',
            action="Walk",
            candidate_states=['{"x":2}', '{"x":3}', '{"x":4}', '{"x":5}'],
            correct_choice=2,
        )
        got = select_rows([row], "minecraft", 1)[0]
        self.assertEqual(got["target"], 2)
        prompt = messages_for(got, "minecraft", "text", "chain-of-thought")[0][
            "content"
        ]
        self.assertIn("Choice 4", prompt)
        self.assertNotIn("correct_choice", prompt)


class ScoringTests(unittest.TestCase):
    def test_chess_per_square_and_invalid(self):
        board = chess.Board()
        initial = board.fen()
        board.push_uci("e2e4")
        target = board.fen()
        good = score_single_prediction_chess(
            target, "FINAL ANSWER: " + target, False, initial
        )
        self.assertEqual(good["successes"], 64)
        unchanged = score_single_prediction_chess(
            target, "FINAL ANSWER: " + initial, False, initial
        )
        self.assertEqual(unchanged["successes"], 62)
        self.assertEqual(unchanged["exact_match"], 0)
        bad = score_single_prediction_chess(target, "not a FEN", False, initial)
        self.assertEqual(bad["successes"], 0)
        self.assertEqual(bad["invalid_pred"], 1)

    def test_final_answers(self):
        self.assertEqual(
            score_single_prediction_shell("2", "Started at 1. FINAL ANSWER: 2")[
                "successes"
            ],
            1,
        )
        self.assertEqual(
            score_single_prediction_minecraft(4, "FINAL ANSWER: 4")["successes"], 1
        )
        self.assertEqual(
            score_single_prediction_minecraft(4, "FINAL ANSWER: 3")["successes"], 0
        )


if __name__ == "__main__":
    unittest.main()
