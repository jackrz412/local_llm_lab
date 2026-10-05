"""Build data/okun.csv: annual real GDP growth and unemployment change, 1960 onward.

Downloads two public FRED series (Federal Reserve Bank of St. Louis):
    GDPC1  - real GDP, quarterly
    UNRATE - civilian unemployment rate (%), monthly

Usage (from the project root):
    python3 scripts/build_okun_data.py
"""

import csv
import sys
import urllib.request
from pathlib import Path

# ---- Settings ---------------------------------------------------------------

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Series ID -> number of observations in a complete year.
SERIES = {"GDPC1": 4, "UNRATE": 12}

FIRST_YEAR = 1960

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/okun.csv")

# ---- Download ---------------------------------------------------------------


def download(series):
    """Save one FRED series to data/raw/<series>.csv. Return True on success."""
    url = FRED_CSV_URL.format(series=series)
    try:
        # Use urllib's default User-Agent: FRED drops requests with unfamiliar ones.
        with urllib.request.urlopen(url, timeout=30) as response:
            (RAW_DIR / f"{series}.csv").write_bytes(response.read())
        print(f"Downloaded {series}")
        return True
    except OSError as error:  # covers network errors and HTTP errors
        print(f"Could not download {series}: {error}")
        return False


def get_raw_files():
    """Download every series, falling back to files already in data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for series in SERIES:
        path = RAW_DIR / f"{series}.csv"
        if not download(series):
            if path.exists():
                print(f"  Using the existing copy at {path}")
            else:
                missing.append(series)

    if missing:
        lines = ["\nPlease download these files by hand and re-run this script:"]
        for series in missing:
            lines.append(f"  1. Open https://fred.stlouisfed.org/series/{series}")
            lines.append("  2. Click 'Download' and choose 'CSV (data)'")
            lines.append(f"  3. Save it as {RAW_DIR / (series + '.csv')}")
        sys.exit("\n".join(lines))


# ---- Read and average -------------------------------------------------------


def read_by_year(series):
    """Return {year: [values...]} from a FRED CSV (columns: date, value)."""
    by_year = {}
    with (RAW_DIR / f"{series}.csv").open() as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        for date, value in reader:
            if value in ("", "."):  # FRED marks missing values with "."
                continue
            year = int(date[:4])
            by_year.setdefault(year, []).append(float(value))
    return by_year


def annual_averages(by_year, per_year):
    """Average each year that has all of its observations; skip partial years."""
    return {
        year: sum(values) / len(values)
        for year, values in by_year.items()
        if len(values) == per_year
    }


# ---- Main -------------------------------------------------------------------


def main():
    get_raw_files()

    raw = {series: read_by_year(series) for series in SERIES}
    averages = {
        series: annual_averages(raw[series], per_year)
        for series, per_year in SERIES.items()
    }
    gdp, unemp = averages["GDPC1"], averages["UNRATE"]

    # Latest year where both series are complete.
    last_year = max(set(gdp) & set(unemp))

    # Report years after last_year that were left out, and why.
    for year in range(last_year + 1, max(raw["UNRATE"]) + 1):
        counts = ", ".join(
            f"{series} has {len(raw[series].get(year, []))}/{per_year}"
            for series, per_year in SERIES.items()
        )
        print(f"Skipping {year} (incomplete: {counts})")

    rows = []
    for year in range(FIRST_YEAR, last_year + 1):
        # Growth needs both this year and the year before.
        if not all(y in gdp and y in unemp for y in (year, year - 1)):
            print(f"Warning: skipping {year}, missing data for it or {year - 1}")
            continue
        gdp_growth = (gdp[year] / gdp[year - 1] - 1) * 100
        unemp_change = unemp[year] - unemp[year - 1]
        rows.append([year, round(gdp_growth, 4), round(unemp_change, 4)])

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "gdp_growth", "unemp_change"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} years ({rows[0][0]}-{rows[-1][0]}) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
