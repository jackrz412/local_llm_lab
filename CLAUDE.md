# Local LLM Lab — project instructions

## What this is
Jack Roberts' benchmark lab: it compares large language models running locally through Ollama on the same tasks.
It is a public portfolio project (GitHub account `jackrz412`) and will be linked from the portfolio site.

## How to work with me
- I'm learning. Before running a command or adding a dependency, say in one or two sentences what it does and why we need it.
- Define technical terms the first time they come up.
- Prefer the simplest approach that works. Don't add frameworks, libraries, or abstractions we don't need yet.
- Make small, focused changes, and pause after each one so I can review it before you continue.
- Don't commit or push unless I ask you to. I want to run Git myself while I'm learning it.

## Stack
- Ollama runs the models locally (installed; models live in `~/.ollama`, outside this repo).
- Models under test (all already pulled):
  - `llama3.2:3b` (2.0 GB)
  - `llama3.1:8b` (4.9 GB)
  - `gemma3:12b` (8.1 GB)
- Language: Python 3.9 (system `python3`).
- Libraries: standard library only for `run_bench.py` and `scripts/build_okun_data.py`
  (talks to Ollama's REST API at `localhost:11434` via `urllib`).
  `pandas` and `statsmodels`, pinned in `requirements.txt` and installed in `.venv/`,
  for `scripts/okun_regression.py` only.
- Data: public FRED series GDPC1 and UNRATE, downloaded to `data/raw/`.

## Hard rules
- Never include anything from my employers or their programs, sanitized or not. Use only public data sources.
- Never put personal contact details (phone, home address, personal email) in the code or in commits.
- Never commit secrets: API keys, tokens, `.env` files.
- Never commit model files; they are too large for GitHub.

## Commands
Run everything from the project root. Ollama must be running for the benchmark.

Setup (once): create the virtual environment and install the packages the Okun scripts need.
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Okun ground-truth pipeline (run in order):
```
python3 scripts/build_okun_data.py          # download FRED data -> data/okun.csv
.venv/bin/python scripts/okun_regression.py # fit regressions -> results/okun_ground_truth.{txt,json}
```

Benchmark (appends to results/results.jsonl):
```
python3 run_bench.py                                 # prompts.json, all models, 3 runs each
python3 run_bench.py --prompts prompts_okun.json     # Okun prompt set
python3 run_bench.py --prompts prompts_okun.json --models llama3.2:3b \
    --only okun-trap --runs 1 --tag smoke-test       # quick smoke test
python3 run_bench.py --help                          # all options
```

`run_bench.py` and `build_okun_data.py` use only the standard library; `okun_regression.py` needs `.venv`.
`prompts_okun.json` embeds data from `data/okun.csv` and the full-sample summary. If the data changes, its prompts and expected answers must be updated too (there is no script for this yet).
