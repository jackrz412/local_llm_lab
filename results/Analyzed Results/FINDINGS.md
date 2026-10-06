# Findings: three local LLMs on general and economics tasks

Three open models were run locally with Ollama on an Apple M4 Mac (16 GB RAM).
Each one answered 12 prompts 3 times, for 108 graded answers.
The runs took place on 5 October 2026.

## Three main takeaways

1. **The largest model was the most accurate and the only consistent one, but it took 5× as long per correct answer.**
   `gemma3:12b` passed 67% of answers and gave the same verdict on all 3 runs of every prompt.
   The two Llama models passed 56% and 44%, and the 3B model's results changed between runs on 4 of 12 prompts.
   Gemma needed 45 seconds per correct answer, against 9 seconds for `llama3.2:3b`.
   Two things drove that: Gemma generates 3.5× fewer tokens per second, and its answers are longer.

2. **A pass/fail score hides why models fail, and a quarter of the fails were only about format.**
   12 of the 48 fails had correct content in the wrong format: JSON wrapped in a Markdown code block.
   Leave out that format rule and the two prompts every model failed by design or by prompt flaw, and Gemma passes **30 of 30**.
   The smaller models' fails were real errors: wrong arithmetic, describing the wrong economic law, broken code and invented data.

3. **The models were reliable at writing code and reporting numbers, and unreliable at calculating numbers or judging the data.**
   On Okun's Law, 7 of 9 generated scripts reproduced the correct regression exactly, and all 9 copied the coefficients correctly into JSON.
   None of the 9 attempts to calculate the regression without code came close.
   None of 9 interpretations noticed that 2020 (COVID) distorts the sample.
   In practice: have a model write the analysis code, and keep a human responsible for judging the data.

---

## Setup

| | |
|---|---|
| Models | `llama3.2:3b` (2.0 GB), `llama3.1:8b` (4.9 GB), `gemma3:12b` (8.1 GB) |
| Prompt sets | **General** (5 prompts): factual recall, arithmetic, summarization, JSON output, trap question<br>**Okun's Law** (7 prompts): recall, applied reasoning, in-head regression, code generation, interpretation, JSON output, trap question |
| Runs | 3 per model and prompt, default Ollama settings |
| Ground truth | OLS regression of the change in unemployment on real GDP growth, using public FRED data for 1960–2024 (series GDPC1 and UNRATE): slope −0.4354, intercept 1.2958, R² 0.614, n = 65 |
| Grading | 99 answers graded by automatic checks in `grade_results.py`, 9 by hand (`okun-recall`). Every fail is given one failure mode, its main cause |

## 1. Token performance

### Generation speed

| Model | Tokens/s, mean | Std dev | Range | Coefficient of variation | First-run load time |
|---|---|---|---|---|---|
| `llama3.2:3b` | **46.1** | 1.43 | 43.3–48.9 | 3.1% | 1–2 s |
| `llama3.1:8b` | **21.3** | 0.46 | 20.3–22.3 | 2.2% | ~5 s |
| `gemma3:12b` | **13.0** | 0.51 | 11.6–13.8 | 4.0% | ~7 s |

- **Speed is very consistent.** Variation stays under 5% for every model, so the differences between models are real rather than noise.
- **Speed follows model size.** Tokens/s × model size is close to constant: 92, 104 and 105 GB/s.
  To produce each token, a model reads essentially all of its weights from memory, so speed is roughly memory bandwidth ÷ model size.
  The M4's bandwidth is about 120 GB/s, and the models used 80–88% of it.
- **Longer inputs and outputs are slightly slower.** Prompts that included 65 rows of data, or produced long answers, ran about 10% slower than one-line questions.
  For example, the 3B model ran at 43.8 tokens/s on those prompts and at 48.5 tokens/s on the factual question.

### Practical speed

Tokens differ in size between model families, so characters per second is the fairer comparison across them.

| Model | Characters per token | Characters/s | Mean tokens per answer | Median time per answer | Longest answer | Total time, 36 answers |
|---|---|---|---|---|---|---|
| `llama3.2:3b` | 3.83 | 176 | 217 | 3.6 s | 21 s | 182 s |
| `llama3.1:8b` | 3.82 | 81 | 220 | 6.8 s | 75 s | 398 s |
| `gemma3:12b` | 2.92 | 38 | 356 | 13.3 s | 142 s | 1,081 s |

- Gemma is 3.5× slower than the 3B model by tokens/s, and **4.6× slower by characters/s**.
  Its tokens hold fewer characters, probably because of its tokenizer and its heavy Markdown formatting.
- Gemma is also the most verbose. On the Okun trap question it averaged 1,250 tokens to conclude "cannot be determined"; the 8B model needed 176.

## 2. Correctness

### Pass rates

| Model | All prompts | General set | Okun set | Without `okun-compute` and `okun-interpret`¹ | Same, with code-block JSON accepted |
|---|---|---|---|---|---|
| `llama3.2:3b` | 20/36 (56%) | 13/15 (87%) | 7/21 (33%) | 20/30 (67%) | 20/30 (67%) |
| `llama3.1:8b` | 16/36 (44%) | 9/15 (60%) | 7/21 (33%) | 16/30 (53%) | 22/30 (73%) |
| `gemma3:12b` | **24/36 (67%)** | 12/15 (80%) | **12/21 (57%)** | **24/30 (80%)** | **30/30 (100%)** |

¹ Every model failed these two prompts on every run. `okun-compute` asks for a regression calculated in the model's head and was designed to fail. `okun-interpret` turned out to be a flawed prompt (see Limitations).

### Scorecard (passes out of 3 runs)

| Prompt | `llama3.2:3b` | `llama3.1:8b` | `gemma3:12b` |
|---|---|---|---|
| factual-01 | 3 | 3 | 3 |
| arithmetic-01 | 1 | 0 | 3 |
| summarization-01 | 3 | 3 | 3 |
| json-01 | 3 | 0 | 0 |
| trap-01 | 3 | 3 | 3 |
| okun-recall | 0 | 0 | 3 |
| okun-reason | 1 | 1 | 3 |
| okun-compute | 0 | 0 | 0 |
| okun-code | 1 | 3 | 3 |
| okun-interpret | 0 | 0 | 0 |
| okun-json | 3 | 0 | 0 |
| okun-trap | 2 | 3 | 3 |

### Consistency across runs

A prompt counts as inconsistent when its 3 runs did not all get the same grade.

- `llama3.2:3b`: **4 of 12** prompts inconsistent (arithmetic, okun-reason, okun-code, okun-trap).
- `llama3.1:8b`: 1 of 12 (okun-reason). It was consistently wrong elsewhere; on the arithmetic problem it gave three different wrong times.
- `gemma3:12b`: **0 of 12**.

With a single run, the 3B model could have looked able, or unable, to do the arithmetic problem, depending on which run was used.

### Time per correct answer

| Model | Correct answers | Total time | Seconds per correct answer |
|---|---|---|---|
| `llama3.2:3b` | 20 | 182 s | **9.1** |
| `llama3.1:8b` | 16 | 398 s | 24.9 |
| `gemma3:12b` | 24 | 1,081 s | 45.0 |

The 8B model came out worst on this measure: it was 2.7× slower than the 3B model without being more accurate.

## 3. Why answers failed

| Failure mode | Meaning | `llama3.2:3b` | `llama3.1:8b` | `gemma3:12b` | Total |
|---|---|---|---|---|---|
| `wrong_answer` | A stated value or calculation is wrong | 7 | 8 | 3 | 18 |
| `format` | Right content, wrong form | – | 6 | 6 | 12 |
| `omission` | Missed a required point | 3 | 3 | 3 | 9 |
| `wrong_concept` | Answered about a different idea | 2 | 1 | – | 3 |
| `contradiction` | Contradicted itself | 1 | 1 | – | 2 |
| `hallucination` | Presented invented facts or data as real | 1 | 1 | – | 2 |
| `broken_code` | Code crashed or gave the wrong result | 2 | – | – | 2 |
| **Total fails** | | **16** | **20** | **12** | **48** |

**What the failure modes show:**
- **Gemma: 6 format fails, plus the 6 shared by every model.** All 6 format fails are JSON wrapped in a code block. Its other 6 are the designed-to-fail regression and the prompt-design omission. None of its fails came from a conceptual, reasoning or factual error.
  Two of its *passing* answers still contained mistakes:
  - On the Okun trap question, it reached the right conclusion but along the way gave a "1960–2024 coefficient" of −0.525 (the true value is −0.4354).
  - It made up today's date (see below).
- **`llama3.1:8b` fails on substance and on format.**
  - It confused Okun's Law with the Phillips curve, which relates unemployment to inflation.
  - It invented an equation and presented it as the "typical estimate".
  - It answered the applied question with +5 and −4 points, where the answer is about +1.
- **`llama3.2:3b` follows format instructions exactly but makes the most varied substantive errors.**
  - One script crashed and another produced the wrong intercept.
  - It described an unrelated production-theory idea as Okun's Law.
  - It invented a 1950s coefficient for data that starts in 1960, presenting the 1960s rows as if they were 1950s data.
- **Every model showed the identical omission.** Each model missed the 2020 outlier in all 3 of its runs. That pattern points to the prompt rather than the models.

**Other observations:**
- **The general-knowledge trap was easy.** Every model correctly said no one has walked on Mars.
- **Gemma still made up dates.** In those answers it stated today's date as fact, and gave two different wrong dates ("October 26, 2023" and "June 6, 2024").
- **The in-head regression failed in different ways.** Gemma at least got the sign right (slopes of −0.10 to −0.25 against −0.4354). The Llama models often got the sign wrong.

## Limitations

- **Small sample.** There were 12 prompts and 3 runs each. The counts describe these runs, not general model ability, and a difference of a few fails is not meaningful.
- **One machine and default settings.** All runs used one Mac and Ollama's default temperature (randomness). Answer lengths, and so the timings, vary between runs.
- **Automatic grading uses heuristics.** The trap and reasoning checks match keywords and patterns. One borderline answer, a hedged trap response from the 3B model, was graded pass.
- **One person graded by hand.** The 9 `okun-recall` grades were not checked by a second reviewer.
- **`okun-interpret` was flawed.** The regression summary shows no individual years, so spotting 2020 required the model to connect "1960–2024" with COVID unprompted. A fairer version would include the data rows.
- **Prompt processing time was not recorded separately.** Estimates from total time suggest 1.7–3.6 s for Gemma on the prompts that include data.

## Reproducing these results

```
python3 scripts/build_okun_data.py
.venv/bin/python scripts/okun_regression.py
python3 run_bench.py
python3 run_bench.py --prompts prompts_okun.json
python3 grade_results.py --run-code
```

Raw answers are in `results/results.jsonl`. Per-answer grades with notes are in `results/Analyzed Results/grades.csv`. Hand grades are in `results/manual_grades.csv`.
