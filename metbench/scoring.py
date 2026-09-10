"""MET-Bench evaluation metrics and response parsers."""

import chess
import re
import sys
from statsmodels.stats.proportion import proportion_confint

from metbench.confidence import clustered_ratio_interval

ALPHA = 0.05
CI_METHOD = "wilson"


def ci_half_width(
    successes: float, total: int, alpha: float = ALPHA, method: str = CI_METHOD
) -> float:
    """
    Compute half‐width of a (1-alpha) Wilson confidence interval
    for a binomial proportion using statsmodels.
    Returns 0.0 if total == 0.
    """
    if total == 0:
        return 0.0
    lo, hi = proportion_confint(successes, total, alpha=alpha, method=method)
    return (hi - lo) / 2


def parse_chess_fen_gt(fen_str):
    """
    Parse a ground truth FEN string (no FINAL ANSWER prefix expected).
    If invalid, return None. If valid, return a chess.Board.
    """
    try:
        fen_str = str(fen_str).strip().split("\n")[0]
        fen_str = fen_str.replace("`", "")
        fen_str = fen_str.replace("*", "")
        return chess.Board(fen=fen_str)
    except Exception as e:
        print(e)
        return None


def _clean_chess_response_text(text):
    text = str(text)
    for token in ("\\boxed{", "\\boxed"):
        text = text.replace(token, " ")
    for char in "`*[](){}'\",;:.":
        text = text.replace(char, " ")
    return text


def _valid_chess_fens(text):
    tokens = _clean_chess_response_text(text).split()
    candidates = []
    for idx in range(len(tokens) - 5):
        candidate = " ".join(tokens[idx : idx + 6])
        try:
            chess.Board(fen=candidate)
        except Exception:
            continue
        candidates.append(candidate)
    return candidates


def _single_chess_fen(text):
    candidates = _valid_chess_fens(text)
    if len(candidates) == 1:
        return candidates[0]
    return None


def parse_chess_fen(fen_str):
    """
    Parse a predicted FEN from model response.
    Accept only bottom-up final-answer lines containing exactly one valid FEN.
    """
    try:
        text = str(fen_str)
        for line in reversed(text.splitlines()):
            if "final answer" not in line.lower():
                continue
            candidate = _single_chess_fen(line)
            if candidate is not None:
                return chess.Board(fen=candidate)
        return None
    except Exception as e:
        print(e)
        return None


def _parse_choice_answer(response, maximum):
    """Accept one explicit final answer or a standalone choice."""
    if not response:
        return None
    text = str(response).strip()
    markers = list(re.finditer(r"final\s+answer\s*:\s*", text, re.IGNORECASE))
    if markers:
        text = text[markers[-1].end() :].strip()
        text = text.splitlines()[0] if text else ""
    text = text.replace(r"\boxed", "")
    text = re.sub(r"[`*{}\[\]]", "", text).strip()
    match = re.fullmatch(
        r"(?:(?:choice|candidate|shell)\s*#?\s*)?([1-" + str(maximum) + r"])[.!]?",
        text,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def parse_shell_arrangement(arr_str):
    """Parse an unambiguous shell number from 1 to 3."""
    return _parse_choice_answer(arr_str, 3)


def _squares_match(board_a, board_b, sq):
    """Return True if both boards have the same piece (or both empty) at sq."""
    pa = board_a.piece_at(sq)
    pb = board_b.piece_at(sq)
    if pa is None and pb is None:
        return True
    if pa is not None and pb is not None and pa.symbol() == pb.symbol():
        return True
    return False


def _is_legal_board(board):
    """Check whether a chess.Board represents a fully legal position."""
    try:
        return board.status() == chess.STATUS_VALID
    except Exception:
        return False


def score_single_prediction_chess(gt_fen, pred_fen, baseline, initial_fen=None):
    """
    Compare two FENs square‐by‐square and compute additional metrics.
    Return dict with:
      - invalid_pred (0 or 1)
      - successes  = # matched squares (0–64)
      - trials     = 64
      - exact_match = 1 if the normalized full FENs match, else 0
      - exact_board_match = 1 if the piece placements match, else 0
      - legal_board = 1 if predicted board is a legal chess position, else 0
      - changed_sq_successes = # correct among squares that changed from initial
      - changed_sq_trials    = # squares that changed from initial position
    """
    invalid_pred = 0
    exact_match = 0
    exact_board_match = 0
    legal_board = 0

    gt_board = parse_chess_fen_gt(gt_fen)
    if gt_board is None:
        print(f"Error: Invalid ground truth FEN: {gt_fen}")
        sys.exit(1)

    # Determine initial board for changed-square metric
    if initial_fen is not None:
        init_board = parse_chess_fen_gt(initial_fen)
    else:
        init_board = chess.Board()  # standard starting position

    # Identify which squares changed from initial to ground truth
    changed_squares = [
        sq for sq in range(64) if not _squares_match(init_board, gt_board, sq)
    ]
    changed_sq_trials = len(changed_squares)

    if not pred_fen:
        invalid_pred = 1
        successes = 0
        exact_match = 0
        legal_board = 0
        changed_sq_successes = 0
    else:
        pred_board = chess.Board() if baseline else parse_chess_fen(pred_fen)
        if pred_board is None:
            invalid_pred = 1
            successes = 0
            exact_match = 0
            legal_board = 0
            changed_sq_successes = 0
        else:
            exact_match = 1 if pred_board.fen() == gt_board.fen() else 0
            exact_board_match = (
                1 if pred_board.board_fen() == gt_board.board_fen() else 0
            )
            successes = sum(
                1 for sq in range(64) if _squares_match(gt_board, pred_board, sq)
            )
            legal_board = 1 if _is_legal_board(pred_board) else 0
            changed_sq_successes = sum(
                1 for sq in changed_squares if _squares_match(gt_board, pred_board, sq)
            )

    return {
        "invalid_pred": invalid_pred,
        "successes": successes,
        "trials": 64,
        "exact_match": exact_match,
        "exact_board_match": exact_board_match,
        "exact_match_trials": 1,
        "legal_board": legal_board,
        "changed_sq_successes": changed_sq_successes,
        "changed_sq_trials": changed_sq_trials,
        "changed_successes": changed_sq_successes,
        "changed_trials": changed_sq_trials,
    }


def score_single_prediction_shell(gt_str, pred_str):
    """
    For shell: success=1 if predicted shell == ground truth, else 0.
    Return dict with:
      - invalid_pred (0 or 1)
      - successes  = 0 or 1
      - trials     = 1
    """
    invalid_pred = 0
    # Ground truth is a simple number, parse it directly
    gt_val = int(gt_str) if gt_str in ("1", "2", "3") else None
    if gt_val is None:
        print(f"Error: Invalid ground truth shell arrangement: {gt_str}")
        sys.exit(1)

    pred_val = parse_shell_arrangement(pred_str) if pred_str else None
    if pred_val is None:
        invalid_pred = 1
        successes = 0
    else:
        successes = 1 if (pred_val == gt_val) else 0

    return {"invalid_pred": invalid_pred, "successes": successes, "trials": 1}


def parse_minecraft_choice(pred_str):
    """Parse an unambiguous candidate number from 1 to 4."""
    return _parse_choice_answer(pred_str, 4)


def score_single_prediction_minecraft(gt_val, pred_str):
    """
    For minecraft: success=1 if predicted choice == ground truth choice, else 0.
    Return dict with:
      - invalid_pred (0 or 1)
      - successes  = 0 or 1
      - trials     = 1
    """
    invalid_pred = 0
    gt_int = int(gt_val)

    pred_val = parse_minecraft_choice(pred_str) if pred_str else None
    if pred_val is None:
        invalid_pred = 1
        successes = 0
    else:
        successes = 1 if (pred_val == gt_int) else 0

    return {
        "invalid_pred": invalid_pred,
        "successes": successes,
        "trials": 1,
    }


def aggregate_scores(results):
    """
    Aggregate:
      - count
      - invalid_pred_count
      - mean accuracy = total_successes / total_trials
      - 95% CI bounds and half-width; Chess uncertainty is across boards
    For chess results that include extra keys, also aggregate:
      - exact_match_accuracy, exact_match_pm
      - changed_sq_accuracy, changed_sq_pm
      - legal_board_rate, legal_board_pm
    """
    total_invalid = sum(r["invalid_pred"] for r in results)
    total_succ = sum(r["successes"] for r in results)
    total_trials = sum(r["trials"] for r in results)
    count = len(results)

    accuracy = (total_succ / total_trials) if total_trials > 0 else 0.0
    is_chess = any(r["trials"] > 1 for r in results)
    if is_chess:
        lower, upper = clustered_ratio_interval(
            [(r["successes"], r["trials"]) for r in results]
        )
    else:
        lower, upper = (
            proportion_confint(total_succ, total_trials, alpha=ALPHA, method=CI_METHOD)
            if total_trials
            else (0.0, 0.0)
        )
    pm = (upper - lower) / 2 if lower is not None else None

    aggregated = {
        "count": count,
        "invalid_pred_count": total_invalid,
        "accuracy": accuracy,
        "accuracy_pm": pm,
        "accuracy_ci_lower": float(lower) if lower is not None else None,
        "accuracy_ci_upper": float(upper) if upper is not None else None,
        "accuracy_ci_method": "normal_across_examples" if is_chess else "wilson",
        "accuracy_ci_unit": "example",
    }

    exact_trials = sum(r.get("exact_match_trials", 0) for r in results)
    if exact_trials > 0:
        exact_matches = sum(r.get("exact_match", 0) for r in results)
        exact_board_matches = sum(r.get("exact_board_match", 0) for r in results)
        aggregated.update(
            {
                "exact_match": exact_matches / exact_trials,
                "exact_match_pm": ci_half_width(exact_matches, exact_trials),
                "exact_match_count": exact_matches,
                "exact_board_match": exact_board_matches / exact_trials,
                "exact_board_match_pm": ci_half_width(
                    exact_board_matches, exact_trials
                ),
                "exact_board_match_count": exact_board_matches,
            }
        )

    changed_trials = sum(r.get("changed_trials", 0) for r in results)
    if changed_trials > 0:
        changed_successes = sum(r.get("changed_successes", 0) for r in results)
        changed_lower, changed_upper = clustered_ratio_interval(
            [
                (r.get("changed_successes", 0), r.get("changed_trials", 0))
                for r in results
            ]
        )
        changed_pm = (
            (changed_upper - changed_lower) / 2 if changed_lower is not None else None
        )
        aggregated.update(
            {
                "changed_square_accuracy": changed_successes / changed_trials,
                "changed_square_accuracy_pm": changed_pm,
                "changed_square_accuracy_ci_lower": changed_lower,
                "changed_square_accuracy_ci_upper": changed_upper,
                "changed_square_successes": changed_successes,
                "changed_square_trials": changed_trials,
            }
        )

    if results and "legal_board" in results[0]:
        n = len(results)
        legal_boards = sum(r.get("legal_board", 0) for r in results)
        aggregated.update(
            {
                "exact_match_accuracy": aggregated.get("exact_match", 0.0),
                "changed_sq_accuracy": aggregated.get("changed_square_accuracy", 0.0),
                "changed_sq_pm": aggregated.get("changed_square_accuracy_pm", 0.0),
                "legal_board_rate": legal_boards / n if n > 0 else 0.0,
                "legal_board_pm": ci_half_width(legal_boards, n),
            }
        )

    return aggregated
