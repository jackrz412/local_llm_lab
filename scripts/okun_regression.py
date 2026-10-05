"""Fit the difference form of Okun's Law on data/okun.csv and save the ground truth.

Model:  unemp_change = intercept + slope * gdp_growth

Fitted twice: on every year, and on every year except 2020 (the COVID outlier).

Usage (from the project root):
    .venv/bin/python scripts/okun_regression.py
"""

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

DATA_FILE = Path("data/okun.csv")
SUMMARY_FILE = Path("results/okun_ground_truth.txt")
JSON_FILE = Path("results/okun_ground_truth.json")


def key_numbers(model):
    """Pull out the numbers the benchmark prompts will be graded against."""
    return {
        "slope": round(float(model.params["gdp_growth"]), 4),
        "intercept": round(float(model.params["Intercept"]), 4),
        "r_squared": round(float(model.rsquared), 4),
        "n": int(model.nobs),
    }


def main():
    data = pd.read_csv(DATA_FILE)
    first, last = data["year"].min(), data["year"].max()

    samples = {
        "full_sample": (f"{first}-{last}", data),
        "excluding_2020": (f"{first}-{last}, excluding 2020", data[data["year"] != 2020]),
    }

    summaries = []
    numbers = {}
    for name, (label, sample) in samples.items():
        # OLS = ordinary least squares, the standard straight-line regression.
        model = smf.ols("unemp_change ~ gdp_growth", data=sample).fit()
        summaries.append(f"=== {name}: {label} ===\n\n{model.summary().as_text()}\n")
        numbers[name] = key_numbers(model)
        print(f"{name:15} {numbers[name]}")

    SUMMARY_FILE.parent.mkdir(exist_ok=True)
    SUMMARY_FILE.write_text("\n\n".join(summaries))
    JSON_FILE.write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"\nSaved {SUMMARY_FILE} and {JSON_FILE}")


if __name__ == "__main__":
    main()
