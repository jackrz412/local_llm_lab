# Local LLM Lab

A benchmark comparing large language models running entirely on my own machine through [Ollama](https://ollama.com/). No cloud APIs, and no data leaves the computer.

Each model answers the same prompts 3 times. Every answer is graded, and every failure is classified by its cause.
Results are measured on an Apple M4 Mac with 16 GB of RAM.

## Models under test

| Model | Download size |
| --- | --- |
| `llama3.2:3b` | 2.0 GB |
| `llama3.1:8b` | 4.9 GB |
| `gemma3:12b` | 8.1 GB |

## What's tested

| Prompt set | Prompts |
| --- | --- |
| **General** (`prompts_general.json`) | Factual recall, arithmetic word problem, one-sentence summary, JSON-only output, trap question with no real answer |
| **Okun's Law** (`prompts_okun.json`) | Recall the law, apply it, compute a regression in the model's head, write regression code, interpret a regression summary, extract values as JSON, trap question about years the data doesn't cover |

The Okun's Law answers are graded against a real regression: the change in US unemployment against real GDP growth, 1960–2024, using public [FRED](https://fred.stlouisfed.org/) data.
That regression gives a slope of **−0.4354** and an R² of **0.614**, which is in the usual textbook range.

## Results

| Model | Tokens/s | Pass rate (36 answers) | Seconds per correct answer |
| --- | --- | --- | --- |
| `llama3.2:3b` | 46.1 | 56% | 9.1 |
| `llama3.1:8b` | 21.3 | 44% | 24.9 |
| `gemma3:12b` | 13.0 | 67% | 45.0 |

**Main takeaways:**

1. **The largest model was the most accurate and the only consistent one, but it took 5× as long per correct answer.** Generation speed follows model size closely, because each token requires reading the whole model from memory.
2. **A quarter of all failures were format-only.** Most were correct JSON wrapped in a Markdown code block. Excluding them, and the two prompts every model failed, Gemma scored 30/30. The smaller models' failures were real errors.
3. **The models were good at writing analysis code (7 of 9 scripts exact) and poor at calculating numbers or judging the data.** None calculated the regression correctly without code, and none noticed that 2020 distorts the sample.

Full write-up, with descriptive statistics, failure modes and limitations: [FINDINGS.md](results/Analyzed%20Results/FINDINGS.md)

## How it works

```
scripts/build_okun_data.py   download FRED data, build annual series      -> data/okun.csv
scripts/okun_regression.py   fit Okun's Law (the ground truth)            -> results/okun_ground_truth.*
run_bench.py                 send each prompt to each model, 3 times      -> results/results.jsonl
grade_results.py             grade answers, classify failures, scorecard  -> results/Analyzed Results/
```

- **Grading.** `grade_results.py` checks each answer automatically: exact values, required keywords, valid JSON, and actually running generated code.
  Answers that need judgment are graded by hand in `results/manual_grades.csv`.
- **Failure modes.** Each failure is given one of eight causes: `format`, `wrong_answer`, `omission`, `wrong_concept`, `contradiction`, `hallucination`, `broken_code` or `no_answer`.

## Running it yourself

Requires Python 3.9+ and [Ollama](https://ollama.com/) installed and running, with the three models pulled (about 15 GB of disk space):

```sh
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull gemma3:12b
```

Then, from the project root:

```sh
# Setup (once): packages for the regression script
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Ground truth
python3 scripts/build_okun_data.py
.venv/bin/python scripts/okun_regression.py

# Benchmark (appends to results/results.jsonl; about 5 and 25 minutes on an M4)
python3 run_bench.py
python3 run_bench.py --prompts prompts_okun.json

# Grade (--run-code executes the model-written regression scripts)
python3 grade_results.py --run-code
```

`run_bench.py` and `grade_results.py` use only the Python standard library. Run either with `--help` to see all options.

## Data

The only data used is public: FRED series [GDPC1](https://fred.stlouisfed.org/series/GDPC1) (real GDP, quarterly) and [UNRATE](https://fred.stlouisfed.org/series/UNRATE) (unemployment rate, monthly), converted to annual averages.
2025 is excluded because the October 2025 unemployment rate was never published.
