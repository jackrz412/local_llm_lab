"""Grade the answers in results/results.jsonl.

Each answer gets one of three grades:
    pass / fail - decided automatically by the check for its prompt ID
    review      - no reliable automatic check; read the answer yourself

Every fail also gets a failure_mode: its main cause, one of FAILURE_MODES below.

Hand grades go in results/manual_grades.csv (columns: timestamp, model, prompt_id,
run, grade, failure_mode, note). A manual grade replaces the automatic one for that
answer. The grader only reads that file; it never changes it.

Writes to results/Analyzed Results/ (both files are overwritten on every run):
    grades.csv     one row per answer: grade, failure mode, who graded it, note,
                   expected answer and the full response, for reviewing in a spreadsheet
    scorecard.txt  pass counts per model and prompt, then failure modes per model;
                   also printed to the screen

Usage:
    python3 grade_results.py              # grade every result tagged "benchmark"
    python3 grade_results.py --run-code   # also run the okun-code scripts (see below)
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---- Settings ---------------------------------------------------------------

RESULTS_FILE = Path("results/results.jsonl")
MANUAL_GRADES_FILE = Path("results/manual_grades.csv")
OUTPUT_DIR = Path("results/Analyzed Results")
GRADES_FILE = OUTPUT_DIR / "grades.csv"
SCORECARD_FILE = OUTPUT_DIR / "scorecard.txt"
OKUN_TRUTH = json.loads(Path("results/okun_ground_truth.json").read_text())["full_sample"]
OKUN_DATA = Path("data/okun.csv")
VENV_PYTHON = Path(".venv/bin/python")

# True: JSON answers must be bare JSON. False: a ```json code block around it is accepted.
STRICT_JSON = True

# Why an answer failed. Every "fail" gets exactly one: its main cause.
FAILURE_MODES = {
    "format": "right content in the wrong form, e.g. JSON inside a code block",
    "wrong_answer": "a stated value or calculation is wrong",
    "omission": "misses a point the answer required",
    "wrong_concept": "answers about a different idea, e.g. the Phillips curve instead of Okun's Law",
    "contradiction": "the answer contradicts itself",
    "hallucination": "presents invented facts, data or formulas as real",
    "broken_code": "code crashes or gives the wrong result",
    "no_answer": "no final answer could be found",
}

# ---- Helpers ----------------------------------------------------------------


def grade(ok, failure_mode, fail_note, pass_note=""):
    """Return (grade, note, failure_mode); failure_mode is blank unless the answer failed."""
    return ("pass", pass_note, "") if ok else ("fail", fail_note, failure_mode)


def numbers_match(answer, expected, tolerance):
    """True if two dicts have the same keys and matching values (numbers within tolerance)."""
    if not isinstance(answer, dict) or set(answer) != set(expected):
        return False
    for key, value in expected.items():
        if isinstance(value, (int, float)) and isinstance(answer[key], (int, float)):
            if abs(answer[key] - value) > tolerance:
                return False
        elif answer[key] != value:
            return False
    return True


def check_json(text, expected):
    """Pass if the answer is a JSON object with the expected values."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    try:
        answer = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return "fail", "not valid JSON", "format"
    if not numbers_match(answer, expected, tolerance=0.0001):
        return "fail", f"wrong values: {answer}", "wrong_answer"
    if fenced and STRICT_JSON:
        return "fail", "correct values, but wrapped in a code block", "format"
    return "pass", "code block accepted" if fenced else "", ""


def last_float(pattern, text):
    """Return the number captured by the last match of pattern, or None."""
    for match in reversed(re.findall(pattern, text, re.I)):
        try:
            return float(match)
        except ValueError:
            continue
    return None


def run_generated_code(text):
    """Run a model-written script on data/okun.csv and compare its coefficients."""
    block = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = block.group(1) if block else text

    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(OKUN_DATA, folder)  # the prompt asks for a file named okun.csv
        (Path(folder) / "generated.py").write_text(code)
        try:
            result = subprocess.run(
                [str(VENV_PYTHON.absolute()), "generated.py"],
                cwd=folder, capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "fail", "script timed out", "broken_code"

    if result.returncode != 0:
        error = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
        return "fail", f"script crashed: {error[:100]}", "broken_code"

    slope = last_float(r"(?m)^gdp_growth\s+(-?[\d.]+)", result.stdout)
    intercept = last_float(r"(?m)^(?:const|Intercept)\s+(-?[\d.]+)", result.stdout)
    if slope is None or intercept is None:
        return "fail", "no gdp_growth and intercept rows in the output", "broken_code"
    ok = (abs(slope - OKUN_TRUTH["slope"]) < 0.0001
          and abs(intercept - OKUN_TRUTH["intercept"]) < 0.0001)
    return grade(ok, "broken_code", f"wrong result: slope {slope}, intercept {intercept}")


# ---- Checks, one per prompt ID ----------------------------------------------
# Each takes the answer text and returns (grade, note, failure_mode).


def check_factual(text):
    return grade(re.search(r"\bAu\b", text) and "79" in text, "wrong_answer", "missing Au or 79")


def check_arithmetic(text):
    answers = re.findall(r"Answer:\W*(\d{1,2}:\d{2}\s*[AP]\.?M)", text, re.I)
    if not answers:
        return "fail", "no 'Answer: <time>' line", "no_answer"
    final = answers[-1].upper().replace(".", "").replace(" ", "")
    return grade(final == "5:35PM", "wrong_answer", f"answered {answers[-1]}")


def check_summary(text):
    sentences = len(re.findall(r"[.!?](?:\s|$)", text.strip()))
    if sentences != 1:
        return "fail", f"{sentences} sentences, not one", "format"
    ok = re.search(r"direction", text, re.I) and re.search(r"distance", text, re.I)
    return grade(ok, "omission", "does not mention both direction and distance")


def check_mars_trap(text):
    denies = re.search(r"\b(no one|nobody|no human|no person)\b|has(n't| not) (yet )?(walked|been)", text, re.I)
    return grade(denies, "hallucination", "did not say no one has walked on Mars")


def check_okun_reason(text):
    # The last stated change, e.g. "\boxed{1}", "increase by 1 percentage point", "a 2% increase".
    patterns = [
        r"\\boxed\{\s*(-?[\d.]+)",
        r"(?:increase|rise)\w*\s+(?:in the unemployment rate\s+)?(?:by|of)\s+(?:approximately |about |roughly )?(-?[\d.]+)",
        r"(-?[\d.]+)\s*(?:%|percentage points?)\s+(?:increase|rise)",
    ]
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            matches.append((match.start(), match.group(1)))
    for _, value in sorted(matches, reverse=True):
        try:
            change = float(value)
        except ValueError:
            continue
        return grade(0.8 <= change <= 1.1, "wrong_answer", f"predicted {change} points")
    return "review", "could not find a predicted change", ""


def check_okun_compute(text):
    slope = last_float(r"Slope:[\s*]*(-?[\d.]+)", text)
    intercept = last_float(r"Intercept:[\s*]*(-?[\d.]+)", text)
    if slope is None or intercept is None:
        return "fail", "no 'Slope:'/'Intercept:' lines", "no_answer"
    ok = (abs(slope - OKUN_TRUTH["slope"]) <= 0.01
          and abs(intercept - OKUN_TRUTH["intercept"]) <= 0.01)
    values = f"slope {slope}, intercept {intercept}"
    return grade(ok, "wrong_answer", values, values)


def check_okun_interpret(text):
    return grade(re.search(r"2020|covid|pandemic", text, re.I), "omission", "did not mention the 2020 outlier")


def check_okun_trap(text):
    refuses = re.search(
        r"(not possible|impossible|cannot|can't|can not|unable to)\b[^.]{0,80}(1950|determin|calculat|estimat)"
        r"|no data (for|from|on)[^.]{0,30}1950",
        text, re.I)
    return grade(refuses, "hallucination", "did not say the 1950s can't be estimated from this data")


CHECKS = {
    "factual-01": check_factual,
    "arithmetic-01": check_arithmetic,
    "summarization-01": check_summary,
    "json-01": lambda text: check_json(text, {"name": "Mars", "moons": 2, "order_from_sun": 4}),
    "trap-01": check_mars_trap,
    "okun-recall": lambda text: ("review", "no automatic check", ""),
    "okun-reason": check_okun_reason,
    "okun-compute": check_okun_compute,
    "okun-code": None,  # handled in main(): depends on --run-code
    "okun-interpret": check_okun_interpret,
    "okun-json": lambda text: check_json(text, {
        "slope": OKUN_TRUTH["slope"],
        "intercept": OKUN_TRUTH["intercept"],
        "r_squared": round(OKUN_TRUTH["r_squared"], 3),  # the summary shows 3 decimals
    }),
    "okun-trap": check_okun_trap,
}

# ---- Main -------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Grade benchmark results.")
    parser.add_argument("--tag", default="benchmark",
                        help="only grade results with this tag (default: benchmark)")
    parser.add_argument("--run-code", action="store_true",
                        help="run okun-code answers (executes model-written code) instead of marking them for review")
    return parser.parse_args()


def load_expected(prompt_files):
    """Return {(prompt_file, prompt_id): expected answer} from the prompt files."""
    expected = {}
    for name in prompt_files:
        path = Path(name)
        if not path.exists():
            print(f"Warning: {name} not found; its expected answers will be blank")
            continue
        for item in json.loads(path.read_text()):
            expected[(name, item["id"])] = item.get("expected", "")
    return expected


def load_manual_grades():
    """Return {(timestamp, model, prompt_id): (grade, note, failure_mode)} from manual_grades.csv."""
    if not MANUAL_GRADES_FILE.exists():
        return {}
    # utf-8-sig also reads files saved by Excel, which may start with a UTF-8 marker.
    with MANUAL_GRADES_FILE.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    manual = {}
    for row in rows:
        where = f"({row['model']}, {row['prompt_id']}, run {row['run']})"
        failure_mode = (row.get("failure_mode") or "").strip()
        if row["grade"] not in ("pass", "fail"):
            sys.exit(f"{MANUAL_GRADES_FILE}: grade must be 'pass' or 'fail', got {row['grade']!r} {where}")
        if row["grade"] == "fail" and failure_mode not in FAILURE_MODES:
            sys.exit(f"{MANUAL_GRADES_FILE}: a fail needs a failure_mode from "
                     f"{', '.join(FAILURE_MODES)}; got {failure_mode!r} {where}")
        if row["grade"] == "pass" and failure_mode:
            sys.exit(f"{MANUAL_GRADES_FILE}: a pass should have a blank failure_mode {where}")
        manual[(row["timestamp"], row["model"], row["prompt_id"])] = (row["grade"], row["note"], failure_mode)
    return manual


def scorecard(rows):
    """Return the scorecard as lines of text: pass counts per prompt, then failure modes per model."""
    models = list(dict.fromkeys(row["model"] for row in rows))
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[(row["prompt_file"], row["prompt_id"], row["model"])][row["grade"]] += 1

    lines = []
    for prompt_file in dict.fromkeys(row["prompt_file"] for row in rows):
        lines.append(f"\n{prompt_file}")
        lines.append(f"  {'prompt':18}" + "".join(f"{m:>16}" for m in models))
        for prompt_id in dict.fromkeys(row["prompt_id"] for row in rows if row["prompt_file"] == prompt_file):
            cells = []
            for model in models:
                c = counts[(prompt_file, prompt_id, model)]
                cell = f"{c['pass']}/{c['pass'] + c['fail']}" if c["pass"] + c["fail"] else ""
                if c["review"]:
                    cell += f" {c['review']} review" if cell else f"{c['review']} review"
                cells.append(cell)
            lines.append(f"  {prompt_id:18}" + "".join(f"{cell:>16}" for cell in cells))

    # Failure modes across all prompt sets: how many fails of each kind per model.
    modes = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["grade"] == "fail":
            modes[row["failure_mode"]][row["model"]] += 1
    lines.append("\nFailure modes (all prompt sets)")
    lines.append(f"  {'failure_mode':18}" + "".join(f"{m:>16}" for m in models))
    for mode in FAILURE_MODES:
        if mode in modes:
            lines.append(f"  {mode:18}" + "".join(f"{modes[mode][m] or '':>16}" for m in models))
    totals = [sum(modes[mode][m] for mode in modes) for m in models]
    lines.append(f"  {'total fails':18}" + "".join(f"{t:>16}" for t in totals))
    return lines


def main():
    args = parse_args()
    results = [json.loads(line) for line in RESULTS_FILE.open()]
    results = [r for r in results if r["tag"] == args.tag]
    expected = load_expected({r["prompt_file"] for r in results})
    manual = load_manual_grades()

    rows = []
    used_manual = set()
    for r in results:
        key = (r["timestamp"], r["model"], r["prompt_id"])
        if key in manual:
            outcome, graded_by = manual[key], "manual"
            used_manual.add(key)
        elif r["prompt_id"] == "okun-code":
            outcome = run_generated_code(r["response"]) if args.run_code else ("review", "re-run with --run-code", "")
            graded_by = "auto"
        elif r["prompt_id"] in CHECKS:
            outcome, graded_by = CHECKS[r["prompt_id"]](r["response"]), "auto"
        else:
            outcome, graded_by = ("review", "no check defined for this prompt ID", ""), "auto"
        rows.append({
            "timestamp": r["timestamp"], "prompt_file": r["prompt_file"], "model": r["model"],
            "prompt_id": r["prompt_id"], "run": r["run"], "grade": outcome[0],
            "failure_mode": outcome[2], "graded_by": graded_by, "note": outcome[1],
            "expected": expected.get((r["prompt_file"], r["prompt_id"]), ""),
            "response": r["response"],
        })

    # A manual grade that matched nothing probably has a typo, e.g. in its timestamp.
    for timestamp, model, prompt_id in sorted(set(manual) - used_manual):
        print(f"Warning: manual grade matched no {args.tag!r} result: {model}, {prompt_id}, {timestamp}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    # utf-8-sig adds a marker that tells Excel the file is UTF-8, so symbols like Δ display correctly.
    with GRADES_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = scorecard(rows)
    print("\n".join(lines))
    SCORECARD_FILE.write_text("\n".join(lines).lstrip("\n") + "\n")

    print(f"\nWrote {len(rows)} grades ({len(used_manual)} manual) to {GRADES_FILE}")
    print(f"Wrote the scorecard to {SCORECARD_FILE}")


if __name__ == "__main__":
    main()
