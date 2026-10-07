"""Export a small summary of the graded benchmark for the portfolio results page.

Reads results/Analyzed Results/grades.csv and results/results.jsonl (benchmark runs only)
and writes results/summary.json. Standard library only.

Usage (from the project root):
    python3 scripts/export_summary.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRADES_CSV = ROOT / "results" / "Analyzed Results" / "grades.csv"
RESULTS_JSONL = ROOT / "results" / "results.jsonl"
GROUND_TRUTH_JSON = ROOT / "results" / "okun_ground_truth.json"
OUT_JSON = ROOT / "results" / "summary.json"

# Not in the data: download sizes from `ollama list`, bandwidth from Apple's M4 spec.
MODELS = [
    {"name": "llama3.2:3b", "size_gb": 2.0},
    {"name": "llama3.1:8b", "size_gb": 4.9},
    {"name": "gemma3:12b", "size_gb": 8.1},
]
HARDWARE = {"machine": "Apple M4 Mac mini, 16 GB RAM", "memory_bandwidth_gb_s": 120}

# Every model failed these on every run (see FINDINGS.md), so the "comparable" rates leave them out.
EXCLUDED_FROM_COMPARABLE = ["okun-compute", "okun-interpret"]

# Same pattern grade_results.py uses to spot JSON wrapped in a Markdown code block.
FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# Quotes chosen for the page. Each excerpt must appear verbatim in that run's response.
QUOTES = [
    {
        "id": "format-json",
        "model": "gemma3:12b",
        "prompt_id": "json-01",
        "caption": "Graded a fail: the JSON is correct, but wrapped in a Markdown code block.",
        "excerpts": [
            {"runs": [1], "text": '```json\n{\n  "name": "Mars",\n  "moons": 2,\n  "order_from_sun": 4\n}\n```'},
        ],
    },
    {
        "id": "phillips-curve",
        "model": "llama3.1:8b",
        "prompt_id": "okun-recall",
        "caption": "Describes the Phillips curve (unemployment vs inflation), not Okun's Law.",
        "excerpts": [
            {"runs": [2], "text": "Okun's Law states that a decrease in unemployment is associated with an increase in inflation, and vice versa, due to the trade-off between employment and price stability."},
        ],
    },
    {
        "id": "invented-1950s",
        "model": "llama3.2:3b",
        "prompt_id": "okun-trap",
        "caption": "The data starts in 1960. Instead of saying so, it invents values for the 1950s.",
        "excerpts": [
            {"runs": [3], "text": "Looking at the data, we see that the unemp_change for 1950 is not available. However, we can calculate it as follows: "},
            {"runs": [3], "text": "unemp_change_1949 = 0.7833\nunemp_change_1951 = 2.8333"},
        ],
    },
    {
        "id": "made-up-date",
        "model": "gemma3:12b",
        "prompt_id": "trap-01",
        "caption": "A passing answer that still states a made-up date as fact, and a different one on run 3.",
        "excerpts": [
            {"runs": [1, 2], "text": "**No one has ever walked on Mars.** As of today (October 26, 2023),"},
            {"runs": [3], "text": "As of today, June 6, 2024, all missions to Mars have been robotic."},
        ],
    },
]


def load_grades():
    # utf-8-sig skips the byte-order mark at the start of grades.csv.
    with open(GRADES_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_runs():
    runs = []
    with open(RESULTS_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["tag"] == "benchmark":
                runs.append(row)
    return runs


def is_fenced_json(text):
    match = FENCED_JSON.fullmatch(text.strip())
    if not match:
        return False
    try:
        json.loads(match.group(1))
    except ValueError:
        return False
    return True


def passed(row, accept_code_blocks):
    """Did this answer pass? Optionally count code-block JSON format fails as passes."""
    if row["grade"] == "pass":
        return True
    if accept_code_blocks and row["failure_mode"] == "format":
        if not is_fenced_json(row["response"]):
            raise SystemExit(f"format fail is not code-block JSON: {row['model']} {row['prompt_id']} run {row['run']}")
        return True
    return False


def pass_rate(rows, accept_code_blocks):
    return {
        "passed": sum(passed(r, accept_code_blocks) for r in rows),
        "total": len(rows),
    }


def prompt_order(grades):
    """Prompt IDs grouped by prompt set, in the order they first appear in grades.csv."""
    prompts = []
    for row in grades:
        entry = {"id": row["prompt_id"], "set": "okun" if "okun" in row["prompt_file"] else "general"}
        if entry not in prompts:
            prompts.append(entry)
    return prompts


def scorecard(grades, prompts, accept_code_blocks):
    """Passes out of 3 runs, per model and prompt, in the same prompt order as `prompts`."""
    counts = Counter((r["model"], r["prompt_id"]) for r in grades if passed(r, accept_code_blocks))
    return {m["name"]: [counts[(m["name"], p["id"])] for p in prompts] for m in MODELS}


def okun_compute_guesses(grades):
    """Slope and intercept each model computed in its head, parsed from the grader's note."""
    guesses = []
    for row in grades:
        if row["prompt_id"] != "okun-compute":
            continue
        match = re.search(r"slope (-?[\d.]+), intercept (-?[\d.]+)", row["note"])
        guesses.append({
            "model": row["model"],
            "run": int(row["run"]),
            "slope": float(match.group(1)) if match else None,
            "intercept": float(match.group(2)) if match else None,
        })
    return guesses


def build_quotes(grades):
    responses = {(r["model"], r["prompt_id"], int(r["run"])): r for r in grades}
    quotes = []
    for quote in QUOTES:
        excerpts = []
        for excerpt in quote["excerpts"]:
            for run in excerpt["runs"]:
                row = responses[(quote["model"], quote["prompt_id"], run)]
                if excerpt["text"] not in row["response"]:
                    raise SystemExit(f"quote {quote['id']} run {run}: excerpt not found verbatim")
            first = responses[(quote["model"], quote["prompt_id"], excerpt["runs"][0])]
            excerpts.append({
                "runs": excerpt["runs"],
                "grade": first["grade"],
                "failure_mode": first["failure_mode"] or None,
                "text": excerpt["text"],
            })
        quotes.append({
            "id": quote["id"],
            "model": quote["model"],
            "prompt_id": quote["prompt_id"],
            "caption": quote["caption"],
            "excerpts": excerpts,
        })
    return quotes


def main():
    grades = load_grades()
    runs = load_runs()
    if len(grades) != len(runs):
        raise SystemExit(f"{len(grades)} grades but {len(runs)} benchmark runs; re-run grade_results.py")

    prompts = prompt_order(grades)
    models = []
    for model in MODELS:
        name = model["name"]
        model_grades = [r for r in grades if r["model"] == name]
        comparable = [r for r in model_grades if r["prompt_id"] not in EXCLUDED_FROM_COMPARABLE]
        model_runs = [r for r in runs if r["model"] == name]
        tokens_per_second = [float(r["tokens_per_second"]) for r in model_runs]
        total_seconds = sum(float(r["total_seconds"]) for r in model_runs)
        correct = sum(passed(r, False) for r in model_grades)
        failure_modes = Counter(r["failure_mode"] for r in model_grades if r["grade"] == "fail")
        models.append({
            "name": name,
            "size_gb": model["size_gb"],
            "tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 1),
            "total_seconds": round(total_seconds, 1),
            "seconds_per_correct": round(total_seconds / correct, 1),
            "pass_rates": {
                "all": {"strict": pass_rate(model_grades, False), "json_accepted": pass_rate(model_grades, True)},
                "comparable": {"strict": pass_rate(comparable, False), "json_accepted": pass_rate(comparable, True)},
            },
            "failure_modes": dict(failure_modes.most_common()),
        })

    with open(GROUND_TRUTH_JSON, encoding="utf-8") as f:
        ground_truth = json.load(f)["full_sample"]

    summary = {
        "hardware": HARDWARE,
        "runs_per_prompt": 3,
        "comparable_excludes": EXCLUDED_FROM_COMPARABLE,
        "models": models,
        "scorecard": {
            "prompts": prompts,
            "strict": scorecard(grades, prompts, False),
            "json_accepted": scorecard(grades, prompts, True),
        },
        "okun_ground_truth": ground_truth,
        "okun_compute_guesses": okun_compute_guesses(grades),
        "quotes": build_quotes(grades),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Print the headline numbers to check against FINDINGS.md.
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    for m in models:
        rates = m["pass_rates"]
        print(
            f"  {m['name']:<12} {m['tokens_per_second']:>5} tok/s  "
            f"all {rates['all']['strict']['passed']}/36  "
            f"comparable {rates['comparable']['strict']['passed']}/30 "
            f"({rates['comparable']['json_accepted']['passed']}/30 json accepted)  "
            f"{m['seconds_per_correct']} s/correct  "
            f"{sum(m['failure_modes'].values())} fails"
        )


if __name__ == "__main__":
    main()
