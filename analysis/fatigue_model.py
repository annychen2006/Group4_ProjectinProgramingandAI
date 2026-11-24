"""Social Battery v1: should the app tell this user to recharge today?

Group model assignment (regression). Loads the team's logs, fits v1, prints
the fit evaluation, values the nudge rule against the baseline, and returns
a decision for a new day.

Data (self-logged by the four team members, 14-19 October 2026):
  data/interactions.csv   one row per interaction (type, minutes, who, date)
  data/daily_fatigue.csv  one row per person per day, with the outcome
                          fatigue_score (1 = fully charged, 10 = drained)

Model: OLS of fatigue_score on total_hours (hours of logged interactions
that day). 8 person-days is under 30, so per the brief there is no 70/30
split: v1 is fit on all 8 rows. As an extra check on new days, each day is
also predicted by a model fit on the other 7 (leave-one-out).

Decision rule: nudge ("your battery is low, block some recharge time") if
predicted fatigue >= TIRED_AT, otherwise stay quiet.
Baseline: predict the average fatigue for every day, so the same decision
every day.

Usage: python analysis/fatigue_model.py
Writes analysis/fatigue_model_report.md, analysis/fatigue_model_results.json
and analysis/fatigue_model_chart.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from regressions import build_daily_table

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

TIRED_AT = 7  # a day scored 7+ is a "tired" day; the app nudges when it predicts 7+

# ASSUMED value per user-day, in dollars (no real figures yet; change these).
VALUE = {
    "hit": 0.50,          # nudged on a tired day: user rests, stays engaged with the app
    "false_alarm": -0.20, # nudged on a fine day: an annoying notification, small churn risk
    "miss": -0.50,        # no nudge on a tired day: user burns out, the app failed its job
    "quiet_ok": 0.00,     # no nudge on a fine day: nothing happens
}
USERS = 1_000     # ASSUMED active users
DAYS_PER_YEAR = 365

# One realistic new day the app might see (minutes by interaction type).
NEW_DAY = {
    "description": "weekday: 2 h lecture, 1.5 h study, 1 h group work, 1.5 h with friends",
    "interactions": [("lecture", 120), ("study", 90), ("group work", 60), ("friend", 90)],
}

FORMULA = "fatigue_score ~ total_hours"


def load():
    table = build_daily_table()
    logs = pd.read_csv(DATA / "daily_fatigue.csv", parse_dates=["date"])
    table = table.merge(logs[["Name", "date", "total_duration_min"]], on=["Name", "date"])
    table["total_hours"] = table["total_duration_min"] / 60
    return table


def r2_rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    sse = float(((actual - predicted) ** 2).sum())
    sst = float(((actual - actual.mean()) ** 2).sum())
    return {"r2": 1 - sse / sst, "rmse": (sse / len(actual)) ** 0.5}


def outcome(actual, predicted):
    tired = np.asarray(actual) >= TIRED_AT
    nudge = np.asarray(predicted) >= TIRED_AT
    counts = {
        "hit": int((nudge & tired).sum()),
        "false_alarm": int((nudge & ~tired).sum()),
        "miss": int((~nudge & tired).sum()),
        "quiet_ok": int((~nudge & ~tired).sum()),
    }
    total = sum(counts[k] * VALUE[k] for k in counts)
    per_day = total / len(tired)
    return {**counts, "nudges": counts["hit"] + counts["false_alarm"], "value_total": total,
            "value_per_user_day": per_day, "value_per_year": per_day * USERS * DAYS_PER_YEAR}


def main():
    t = load()
    y = t["fatigue_score"]
    print("Data (one row per person per day):")
    print(t[["Name", "date", "fatigue_score", "interaction_count", "total_hours"]]
          .assign(date=t["date"].dt.strftime("%a %d %b")).round(2).to_string(index=False), "\n")

    v1 = smf.ols(FORMULA, data=t).fit()
    print(v1.summary(), "\n")
    average = float(y.mean())

    def predict_fatigue(total_hours):
        """v1's predicted fatigue score for a day with this many hours of interactions."""
        return float(v1.params["Intercept"] + v1.params["total_hours"] * total_hours)

    def should_nudge(day_interactions):
        """The app's rule for one day: list of (type, minutes) -> 'nudge' or 'stay quiet'."""
        hours = sum(minutes for _, minutes in day_interactions) / 60
        return "nudge" if predict_fatigue(hours) >= TIRED_AT else "stay quiet"

    # Leave-one-out: predict each day from a model fit on the other 7.
    loo_v1, loo_base = [], []
    for i in t.index:
        rest = t.drop(i)
        loo_v1.append(float(smf.ols(FORMULA, data=rest).fit().predict(t.loc[[i]]).iloc[0]))
        loo_base.append(float(rest["fatigue_score"].mean()))

    fitted = v1.fittedvalues
    fit = {
        "baseline_all_days": r2_rmse(y, [average] * len(y)),
        "v1_all_days": r2_rmse(y, fitted),
        "baseline_leave_one_out": r2_rmse(y, loo_base),
        "v1_leave_one_out": r2_rmse(y, loo_v1),
    }
    decisions = {
        "baseline": outcome(y, [average] * len(y)),
        "v1": outcome(y, fitted),
        "v1_leave_one_out": outcome(y, loo_v1),
    }
    hours_threshold = (TIRED_AT - v1.params["Intercept"]) / v1.params["total_hours"]

    new_hours = sum(m for _, m in NEW_DAY["interactions"]) / 60
    new_day = {**NEW_DAY, "total_hours": new_hours,
               "predicted_fatigue": predict_fatigue(new_hours),
               "decision_v1": should_nudge(NEW_DAY["interactions"]),
               "decision_baseline": "nudge" if average >= TIRED_AT else "stay quiet"}

    t["predicted_v1"] = fitted
    t["nudge_v1"] = np.where(fitted >= TIRED_AT, "nudge", "stay quiet")
    results = {
        "rows": len(t), "people": sorted(t["Name"].unique().tolist()),
        "period": [t["date"].min().strftime("%Y-%m-%d"), t["date"].max().strftime("%Y-%m-%d")],
        "interactions_logged": int(len(pd.read_csv(DATA / "interactions.csv"))),
        "split": "none: 8 rows < 30, v1 fit on all rows (leave-one-out used as a held-out check)",
        "formula": FORMULA,
        "coefficients": {k: {"coef": float(v1.params[k]), "p_value": float(v1.pvalues[k])} for k in v1.params.index},
        "r2": float(v1.rsquared), "adj_r2": float(v1.rsquared_adj),
        "average_fatigue": average, "tired_at": TIRED_AT, "tired_days": int((y >= TIRED_AT).sum()),
        "nudge_when_hours_at_least": float(hours_threshold),
        "fit": fit, "value_assumptions": VALUE, "users_assumed": USERS,
        "decisions": decisions, "new_day": new_day,
        "per_day": t[["Name", "date", "total_hours", "fatigue_score", "predicted_v1", "nudge_v1"]]
        .assign(date=t["date"].dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
    }
    (HERE / "fatigue_model_results.json").write_text(json.dumps(results, indent=2))

    # Chart: fatigue vs hours, v1 line vs the baseline's flat average.
    ink, muted, grid, mark, accent = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.linspace(0, 8, 50)
    ax.axhspan(TIRED_AT, 10.5, color="#f4f3f0", zorder=0)
    ax.text(0.15, 10.2, f"tired day ({TIRED_AT}+): app should nudge", color=muted, fontsize=9, va="top")
    ax.plot(xs, [average] * len(xs), color=muted, lw=2, ls="--", zorder=1)
    ax.text(8, average - 0.25, f"baseline: average {average:.2f}, never nudges", color=muted,
            fontsize=9, ha="right", va="top")
    ax.plot(xs, [predict_fatigue(x) for x in xs], color=accent, lw=2, zorder=2)
    ax.text(8, predict_fatigue(8) + 0.2, "v1", color=ink, fontsize=10, ha="right", va="bottom", weight="bold")
    ax.scatter(t["total_hours"], y, s=60, color=mark, edgecolor="white", linewidth=1.5, zorder=3)
    left = {("Taeyun", "Wed"), ("Taeyun", "Sun")}  # labels that would collide with a neighbour
    for _, r in t.iterrows():
        on_left = (r["Name"], f"{r['date']:%a}") in left
        ax.annotate(f"{r['Name']} {r['date']:%a}", (r["total_hours"], r["fatigue_score"]),
                    textcoords="offset points", xytext=(-7 if on_left else 7, -3),
                    ha="right" if on_left else "left", fontsize=8.5, color=muted)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("Hours of interactions that day", color=muted)
    ax.set_ylabel("Fatigue score (1-10)", color=muted)
    ax.set_title(f"Fatigue vs hours of interactions, {len(t)} logged days", loc="left", color=ink, fontsize=12)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(HERE / "fatigue_model_chart.png", dpi=200, facecolor="white")

    # Report.
    usd = lambda v: f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    usd0 = lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    c = results["coefficients"]
    L = ["# Social Battery v1: fatigue model results", "",
         "## Data", "",
         f"- Self-logged by the team ({', '.join(results['people'])}), {results['period'][0]} to {results['period'][1]}.",
         f"- `interactions.csv`: {results['interactions_logged']} interactions. `daily_fatigue.csv`: {len(t)} person-days (the outcome).",
         f"- {len(t)} rows is under 30, so no 70/30 split: v1 is fit on all {len(t)} rows. "
         "Leave-one-out (predict each day from the other 7) is reported as a check on new days.", "",
         "| Person | Date | Hours of interactions | Fatigue | v1 predicted | v1 decision |", "|---|---|---|---|---|---|"]
    for r in results["per_day"]:
        L.append(f"| {r['Name']} | {r['date']} | {r['total_hours']:.2f} | {r['fatigue_score']} | {r['predicted_v1']:.2f} | {r['nudge_v1']} |")
    L += ["", "## Model v1", "", f"`{FORMULA}` (OLS)", "",
          "| Term | Coefficient | p-value |", "|---|---|---|"]
    for k, v in c.items():
        L.append(f"| {k} | {v['coef']:.3f} | {v['p_value']:.3f} |")
    L += ["", f"R² {v1.rsquared:.3f}, adjusted R² {v1.rsquared_adj:.3f}. Each extra hour of interactions adds "
              f"{c['total_hours']['coef']:.2f} fatigue points. v1 predicts a tired day ({TIRED_AT}+) once a day "
              f"reaches {hours_threshold:.1f} hours of interactions.", "",
          "## Fit, baseline vs v1", "",
          "| | R² | Typical error (RMSE, fatigue points) |", "|---|---|---|",
          f"| Baseline (average {average:.2f}), all {len(t)} days | {fit['baseline_all_days']['r2']:.3f} | {fit['baseline_all_days']['rmse']:.2f} |",
          f"| v1, all {len(t)} days | {fit['v1_all_days']['r2']:.3f} | {fit['v1_all_days']['rmse']:.2f} |",
          f"| Baseline, leave-one-out | {fit['baseline_leave_one_out']['r2']:.3f} | {fit['baseline_leave_one_out']['rmse']:.2f} |",
          f"| v1, leave-one-out | {fit['v1_leave_one_out']['r2']:.3f} | {fit['v1_leave_one_out']['rmse']:.2f} |", "",
          "![chart](fatigue_model_chart.png)", "",
          "## Decision and value", "",
          f"Rule: nudge if predicted fatigue >= {TIRED_AT}, else stay quiet (`should_nudge`). "
          f"{results['tired_days']} of {len(t)} logged days were tired days.", "",
          f"Value per user-day (ASSUMED): {VALUE}. Per year: x {USERS:,} users (ASSUMED) x {DAYS_PER_YEAR} days.", "",
          "| Rule | Nudges | Tired days caught | False alarms | Tired days missed | Value, 8 days | Per user-day | Per year |",
          "|---|---|---|---|---|---|---|---|"]
    for label, key in [("Baseline", "baseline"), ("v1", "v1"), ("v1, leave-one-out", "v1_leave_one_out")]:
        d = decisions[key]
        L.append(f"| {label} | {d['nudges']} | {d['hit']} | {d['false_alarm']} | {d['miss']} | {usd(d['value_total'])} | "
                 f"{usd(d['value_per_user_day'])} | {usd0(d['value_per_year'])} |")
    L += ["", "## New input", "",
          f"{NEW_DAY['description']} = {new_hours:.1f} hours. v1 predicts fatigue {new_day['predicted_fatigue']:.2f}: "
          f"**{new_day['decision_v1']}**. Baseline: {new_day['decision_baseline']}.", ""]
    (HERE / "fatigue_model_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
