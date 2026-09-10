#!/usr/bin/env python3
"""Run MET-Bench evaluations, prepare selections, or rescore saved responses."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit
from metbench.data import RELEASES, fingerprint, load_examples, messages_for
from metbench.scoring import (
    aggregate_scores,
    score_single_prediction_chess,
    score_single_prediction_shell,
    score_single_prediction_minecraft,
)


def score(domain, target, response, initial):
    """Score a response using the metrics for its domain."""
    if domain == "chess":
        return score_single_prediction_chess(target, response, False, initial)
    if domain == "shell":
        return score_single_prediction_shell(str(target), response)
    return score_single_prediction_minecraft(target, response)


def summarize(rows):
    """Aggregate completed responses and count errors and truncated outputs."""
    good = [r for r in rows if not r.get("error")]
    result = aggregate_scores(
        [
            score(r["domain"], r["target"], r["response"], r["initial_state"])
            for r in good
        ]
    )
    result.update(
        completed=len(good),
        errors=len(rows) - len(good),
        truncated=sum(r.get("finish_reason") == "length" for r in good),
    )
    return result


def selection_manifest(domain, examples):
    """Record the dataset revision and ordered identifiers of selected examples."""
    repo, revision = RELEASES[domain]
    return dict(
        dataset=repo,
        revision=revision,
        split="test",
        actions=None if domain == "minecraft" else len(examples[0]["actions"]),
        examples=[
            {k: x[k] for k in ["example_id", "source_row", "fingerprint"]}
            for x in examples
        ],
    )


def write_json(path, obj):
    """Replace a JSON file atomically after writing its complete contents."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def read_predictions(path, repair_trailing=False):
    """Read saved responses, recovering an interrupted final write on resume."""
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_trailing or index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            with path.open("r+b") as stream:
                stream.truncate(offset)
            return rows
        offset += len(line)
    if repair_trailing and data and not data.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def run(args):
    """Evaluate the selected tasks, saving responses incrementally for resume."""
    from metbench.backends import APIBackend, HFBackend

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        config = {
            k: v
            for k, v in vars(args).items()
            if k not in ["resume", "output", "command"]
        }
        config["datasets"] = RELEASES
        config["code_sha256"] = hashlib.sha256(
            b"".join(
                p.read_bytes()
                for p in sorted(
                    [Path(__file__)]
                    + list(Path(__file__).parent.joinpath("metbench").glob("*.py"))
                )
            )
        ).hexdigest()
        config = json.loads(json.dumps(config))
        config_path = output / "config.json"
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text()) != config:
                raise ValueError(
                    "Output already exists. Use a new directory or --resume with the same configuration."
                )
        else:
            if args.resume:
                raise ValueError("No run to resume in this directory")
            write_json(config_path, config)
        backend = APIBackend(args) if args.backend == "api" else HFBackend(args)
        all_summaries = {}
        failed = False
        for domain in args.domains:
            examples = load_examples(domain, args.limit, args.actions)
            write_json(
                output / f"{domain}-selection.json",
                selection_manifest(domain, examples),
            )
            for modality in args.modalities:
                prompt = (
                    args.prompt
                    if args.prompt != "paper"
                    else ("chain-of-thought" if domain == "minecraft" else "zero-shot")
                )
                path = output / f"{domain}-{modality}.jsonl"
                previous = {}
                if args.resume and path.exists():
                    for r in read_predictions(path, repair_trailing=True):
                        if not r.get("error"):
                            previous[r["example_id"]] = r
                stop = threading.Event()

                def one(example):
                    """Generate one response and record failures without exposing request contents."""
                    r = dict(
                        domain=domain,
                        modality=modality,
                        prompt=prompt,
                        example_id=example["example_id"],
                        source_row=example["source_row"],
                        fingerprint=example["fingerprint"],
                        target=example["target"],
                        initial_state=example["initial_state"],
                    )
                    if stop.is_set():
                        return dict(r, error="Aborted")
                    try:
                        r.update(
                            backend.generate(
                                messages_for(example, domain, modality, prompt)
                            )
                        )
                    except Exception as exc:
                        # Save the exception type, never credentials or request bodies.
                        r.update(
                            error=type(exc).__name__,
                            status_code=getattr(exc, "status_code", None),
                        )
                        if args.backend == "hf" and isinstance(exc, ValueError):
                            r["error_detail"] = str(exc)[:500]
                        if r["status_code"] in (400, 401, 402, 403, 404):
                            stop.set()
                    return r

                pending = [e for e in examples if e["example_id"] not in previous]
                rows = list(previous.values())
                with (
                    path.open("a") as out,
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.workers
                    ) as pool,
                ):
                    for row in pool.map(one, pending):
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        rows.append(row)
                        print(
                            f"{domain}/{modality}: {len(rows)}/{len(examples)}"
                            + (" ERROR " + row["error"] if row.get("error") else ""),
                            flush=True,
                        )
                summary = summarize(rows)
                summary["expected"] = len(examples)
                summary["status"] = (
                    "complete"
                    if summary["completed"] == len(examples)
                    else "incomplete"
                )
                failed |= bool(summary["errors"])
                all_summaries[f"{domain}/{modality}"] = summary
                write_json(output / "summary.json", all_summaries)
                ci_lower, ci_upper = (
                    summary["accuracy_ci_lower"],
                    summary["accuracy_ci_upper"],
                )
                ci_text = (
                    f"95% CI {ci_lower:.2%}–{ci_upper:.2%}"
                    if ci_lower is not None
                    else "95% CI unavailable: fewer than two examples"
                )
                print(
                    f"{domain}/{modality}: {summary['accuracy']:.2%} ({ci_text}); {summary['errors']} errors",
                    flush=True,
                )
                if stop.is_set():
                    print(
                        "Stopped after a non-retryable API error; inspect the saved status code.",
                        file=sys.stderr,
                    )
                    return 1
            del examples
        return 1 if failed else 0
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    """Parse and validate command-line options, then dispatch the requested command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "prepare"]:
        p = commands.add_parser(command)
        p.add_argument(
            "--domains", nargs="+", choices=list(RELEASES), default=list(RELEASES)
        )
        p.add_argument("--limit", type=int, default=500)
        p.add_argument("--actions", type=int, default=10)
        p.add_argument("--output", required=True)
        if command == "prepare":
            continue
        p.add_argument("--backend", choices=["api", "hf"], default="api")
        p.add_argument("--model", required=True)
        p.add_argument(
            "--modalities",
            nargs="+",
            choices=["text", "image"],
            default=["text", "image"],
        )
        p.add_argument(
            "--prompt",
            choices=["paper", "zero-shot", "chain-of-thought"],
            default="paper",
        )
        p.add_argument("--base-url", default="https://api.openai.com/v1")
        p.add_argument("--api-key-env", default="OPENAI_API_KEY")
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument(
            "--token-parameter",
            choices=["max_tokens", "max_completion_tokens"],
            default="max_completion_tokens",
        )
        p.add_argument(
            "--temperature",
            default="0",
            help="Number, or omit for endpoints that disallow it",
        )
        p.add_argument("--reasoning-effort", default=None)
        p.add_argument("--timeout", type=float, default=180)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--hf-task", choices=["text", "vlm"], default="vlm")
        p.add_argument("--model-revision", default="main")
        p.add_argument(
            "--processor-kwargs",
            default="{}",
            help="JSON options passed to AutoProcessor.from_pretrained",
        )
        p.add_argument("--device-map", default="auto")
        p.add_argument("--resume", action="store_true")
    p = commands.add_parser("score")
    p.add_argument("predictions", type=Path)
    args = parser.parse_args(argv)
    if args.command == "score":
        # Keep the latest attempt for each example when a run was resumed.
        rows = {}
        for row in read_predictions(args.predictions):
            rows[row["example_id"]] = row
        print(json.dumps(summarize(list(rows.values())), indent=2))
        return 0
    if args.limit < 1 or not 1 <= args.actions <= 100:
        parser.error("--limit must be positive and --actions must be 1–100")
    if args.command == "prepare":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for domain in args.domains:
            examples = load_examples(domain, args.limit, args.actions)
            write_json(
                output / f"{domain}-selection.json",
                selection_manifest(domain, examples),
            )
            print(
                domain,
                len(examples),
                "unique examples; scanned through test row",
                examples[-1]["source_row"],
                flush=True,
            )
        return 0
    try:
        if not isinstance(json.loads(args.processor_kwargs), dict):
            raise ValueError()
    except ValueError:
        parser.error("--processor-kwargs must be a JSON object")
    if args.workers < 1 or args.max_tokens < 1:
        parser.error("workers and max-tokens must be positive")
    if args.temperature != "omit":
        try:
            if not 0 <= float(args.temperature) <= 2:
                raise ValueError()
        except ValueError:
            parser.error("--temperature must be between 0 and 2, or omit")
    url = urlsplit(args.base_url)
    if url.username or url.password or url.query:
        parser.error("Pass credentials through the API key environment variable")
    if args.backend == "hf" and (
        args.workers != 1 or (args.hf_task == "text" and "image" in args.modalities)
    ):
        parser.error("HF requires --workers 1; text models require --modalities text")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
