"""Fit every combination of covariates to predict daily fatigue_score.

Covariates (one row per person per day):
  interaction_count  number of interactions logged that day
  academic_pct       academic minutes / 1440 (minutes in a day) x 100
  work_pct           work minutes / 1440 x 100
  social_pct         social minutes / 1440 x 100
  weekend            1 for Saturday/Sunday, 0 for Monday-Friday

Each non-empty subset of covariates is its own OLS regression (with an
intercept). Results are ranked from highest to lowest R^2.

Usage: python analysis/regressions.py
Writes analysis/regression_results.csv (one row per coefficient per model).
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MINUTES_PER_DAY = 24 * 60
ALPHA = 0.05  # significance level for the n_significant / significant_covariates columns

CATEGORIES = {
    "group work": "academic",
    "study": "academic",
    "lecture": "academic",
    "class discussion": "academic",
    "teacher": "academic",
    "work meeting": "work",
    "friend": "social",
    "family": "social",
}
COVARIATES = ["interaction_count", "academic_pct", "work_pct", "social_pct", "weekend"]


def build_daily_table():
    interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["date"])
    fatigue = pd.read_csv(DATA / "daily_fatigue.csv", parse_dates=["date"])

    kind = interactions["type"].str.strip().str.lower()
    unknown = sorted(set(kind) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Add these interaction types to CATEGORIES first: {unknown}")
    interactions["category"] = kind.map(CATEGORIES)

    keys = ["Name", "date"]
    minutes = (
        interactions.pivot_table(
            index=keys, columns="category", values="duration_min", aggfunc="sum", fill_value=0
        )
        .reindex(columns=["academic", "work", "social"], fill_value=0)
    )
    daily = (minutes / MINUTES_PER_DAY * 100).add_suffix("_pct")
    daily["interaction_count"] = interactions.groupby(keys).size()

    table = fatigue[keys + ["fatigue_score"]].merge(daily.reset_index(), on=keys, how="left")
    table[["academic_pct", "work_pct", "social_pct", "interaction_count"]] = table[
        ["academic_pct", "work_pct", "social_pct", "interaction_count"]
    ].fillna(0)
    table["weekend"] = (table["date"].dt.dayofweek >= 5).astype(int)
    return table


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def main():
    table = build_daily_table()
    print("Data used (one row per person per day):")
    print(table.assign(date=table["date"].dt.strftime("%a %Y-%m-%d")).round(2).to_string(index=False))
    print()

    models = []
    for size in range(1, len(COVARIATES) + 1):
        for combo in combinations(COVARIATES, size):
            X = sm.add_constant(table[list(combo)], has_constant="add")
            fit = sm.OLS(table["fatigue_score"], X).fit()
            models.append((combo, fit))
    models.sort(key=lambda item: item[1].rsquared, reverse=True)

    summary = pd.DataFrame(
        {
            "rank": range(1, len(models) + 1),
            "covariates": [" + ".join(combo) for combo, _ in models],
            "n_covariates": [len(combo) for combo, _ in models],
            "R^2": [fit.rsquared for _, fit in models],
            "adj R^2": [fit.rsquared_adj for _, fit in models],
            "F-test p": [fit.f_pvalue for _, fit in models],
        }
    )
    # Covariates (not the intercept) with p < ALPHA in each model.
    significant = [
        [name for name in combo if fit.pvalues[name] < ALPHA] for combo, fit in models
    ]
    summary["n_significant"] = [len(names) for names in significant]
    summary["significant_covariates"] = [", ".join(names) or "-" for names in significant]
    print(f"Ranking of {len(models)} regressions by R^2 (significant = p < {ALPHA}):")
    print(summary.round(3).to_string(index=False))
    print()
    summary.to_csv(HERE / "regression_ranking.csv", index=False)

    rows = []
    print(f"{len(models)} regressions of fatigue_score, ranked by R^2 (n = {len(table)} days)")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1\n")
    for rank, (combo, fit) in enumerate(models, start=1):
        f_p = fit.f_pvalue
        print(
            f"#{rank:<2} {' + '.join(combo)}\n"
            f"    R^2 = {fit.rsquared:.3f}   adj R^2 = {fit.rsquared_adj:.3f}   "
            f"overall F-test p = {f_p:.3f}   residual df = {int(fit.df_resid)}"
        )
        for name in fit.params.index:
            coef, se, p = fit.params[name], fit.bse[name], fit.pvalues[name]
            print(f"      {name:<18} {coef:>9.3f}  (SE {se:.3f})  p = {p:.3f} {stars(p)}")
            rows.append(
                {
                    "rank": rank,
                    "model": " + ".join(combo),
                    "n_covariates": len(combo),
                    "r2": fit.rsquared,
                    "adj_r2": fit.rsquared_adj,
                    "model_f_pvalue": f_p,
                    "term": name,
                    "coef": coef,
                    "std_err": se,
                    "p_value": p,
                }
            )
        print()

    out = HERE / "regression_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved coefficient table to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
