"""Benchmark local Ollama models on a file of prompts.

Every model answers every prompt several times (3 by default). Each answer is
appended as one JSON line to results/results.jsonl, tagged with the prompt file
it came from and a run tag, so different prompt sets and test runs can share it.

Usage:
    python3 run_bench.py                                   # prompts.json, all models
    python3 run_bench.py --prompts prompts_okun.json
    python3 run_bench.py --prompts prompts_okun.json --models llama3.2:3b \\
        --only okun-trap --runs 1 --tag smoke-test

Requires Ollama to be running (the Ollama app, or `ollama serve`).
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---- Settings ---------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

# Default models; override with --models.
MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "gemma3:12b",
]

RUNS_PER_PAIR = 3

RESULTS_FILE = Path("results/results.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models.")
    parser.add_argument("--prompts", default="prompts.json",
                        help="prompt file to run (default: prompts.json)")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="models to test (default: all three)")
    parser.add_argument("--only", nargs="+", metavar="PROMPT_ID",
                        help="run only these prompt IDs (default: all)")
    parser.add_argument("--runs", type=int, default=RUNS_PER_PAIR,
                        help=f"runs per model/prompt pair (default: {RUNS_PER_PAIR})")
    parser.add_argument("--tag", default="benchmark",
                        help="label saved with each result, e.g. smoke-test (default: benchmark)")
    return parser.parse_args()

# ---- Talking to Ollama ------------------------------------------------------


def ask_model(model, prompt):
    """Send one prompt to one model and return Ollama's JSON reply as a dict."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def ns_to_seconds(nanoseconds):
    """Ollama reports durations in nanoseconds (billionths of a second)."""
    return nanoseconds / 1e9


# ---- Main loop --------------------------------------------------------------


def main():
    args = parse_args()
    prompt_file = Path(args.prompts)
    prompts = json.loads(prompt_file.read_text())

    if args.only:
        unknown = set(args.only) - {item["id"] for item in prompts}
        if unknown:
            sys.exit(f"Unknown prompt ID(s) in {prompt_file}: {', '.join(sorted(unknown))}")
        prompts = [item for item in prompts if item["id"] in args.only]

    RESULTS_FILE.parent.mkdir(exist_ok=True)

    total_runs = len(args.models) * len(prompts) * args.runs
    print(f"{len(args.models)} models x {len(prompts)} prompts x {args.runs} runs "
          f"= {total_runs} runs  [{prompt_file.name}, tag: {args.tag}]\n")

    for model in args.models:
        for item in prompts:
            for run in range(1, args.runs + 1):
                print(f"{model} | {item['id']} | run {run}/{args.runs}")

                try:
                    reply = ask_model(model, item["prompt"])
                except urllib.error.URLError as error:
                    sys.exit(f"Could not reach Ollama at {OLLAMA_URL}. Is it running?\n{error}")

                tokens = reply["eval_count"]
                generation_seconds = ns_to_seconds(reply["eval_duration"])

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tag": args.tag,
                    "prompt_file": prompt_file.name,
                    "model": model,
                    "prompt_id": item["id"],
                    "run": run,
                    "response": reply["response"],
                    "tokens_generated": tokens,
                    "tokens_per_second": round(tokens / generation_seconds, 2),
                    "load_seconds": round(ns_to_seconds(reply["load_duration"]), 3),
                    "total_seconds": round(ns_to_seconds(reply["total_duration"]), 3),
                }

                # "a" = append mode: add to the end of the file, never overwrite.
                with RESULTS_FILE.open("a") as f:
                    f.write(json.dumps(record) + "\n")

                print(f"    {tokens} tokens, {record['tokens_per_second']} tok/s, "
                      f"load {record['load_seconds']}s")

    print(f"\nDone. Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
