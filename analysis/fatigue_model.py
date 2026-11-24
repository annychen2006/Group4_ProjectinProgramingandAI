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

# The team spreadsheet's candidate models (daily_fatigue_regression.xlsx), rebuilt
# with its definitions: day share = minutes / 1,440; "class discussion" is not in
# the category mapping, so its minutes are left out of the shares; weekend = Sat/Sun.
SHEET_CATEGORIES = {
    "group work": "academic", "study": "academic", "lecture": "academic", "teacher": "academic",
    "work meeting": "work", "friend": "social", "family": "social",
}
SHEET_MODELS = {
    "M1 Count only": "interaction_count",
    "M2 Time shares only": "social_day_share + academic_day_share + work_day_share",
    "M3 Calendar only": "weekend_indicator",
    "M4 Interaction features": "interaction_count + social_day_share + academic_day_share + work_day_share",
    "M5 Full model": "interaction_count + social_day_share + academic_day_share + work_day_share + weekend_indicator",
}


def load():
    table = build_daily_table()
    logs = pd.read_csv(DATA / "daily_fatigue.csv", parse_dates=["date"])
    table = table.merge(logs[["Name", "date", "total_duration_min"]], on=["Name", "date"])
    table["total_hours"] = table["total_duration_min"] / 60
    return add_sheet_features(table)


def add_sheet_features(table):
    """Add the spreadsheet's features: day shares, recalculated count, weekend indicator."""
    interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["date"])
    interactions["category"] = interactions["type"].str.strip().str.lower().map(SHEET_CATEGORIES)
    keys = ["Name", "date"]
    minutes = interactions.pivot_table(index=keys, columns="category", values="duration_min",
                                       aggfunc="sum", fill_value=0)
    shares = (minutes.reindex(columns=["social", "academic", "work"], fill_value=0) / 1440).add_suffix("_day_share")
    shares["interaction_count"] = interactions.groupby(keys).size()
    table = table.drop(columns=["interaction_count"]).merge(shares.reset_index(), on=keys, how="left")
    share_cols = ["social_day_share", "academic_day_share", "work_day_share", "interaction_count"]
    table[share_cols] = table[share_cols].fillna(0)
    table["weekend_indicator"] = (table["date"].dt.dayofweek >= 5).astype(int)
    return table


def leave_one_out(formula, table):
    """Predict each day from a model fit on the other days."""
    return [float(smf.ols(formula, data=table.drop(i)).fit().predict(table.loc[[i]]).iloc[0])
            for i in table.index]


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

    # Candidate models: the team spreadsheet's M1-M5 and v1, scored on all days and on held-out days.
    candidates = {}
    for name, rhs in {**SHEET_MODELS, "v1 Total hours": "total_hours"}.items():
        formula = f"fatigue_score ~ {rhs}"
        m = smf.ols(formula, data=t).fit()
        held_out = leave_one_out(formula, t)
        candidates[name] = {
            "features": rhs, "parameters": int(len(m.params)), "residual_df": int(m.df_resid),
            "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "f_test_p": float(m.f_pvalue),
            "rmse_all_days": r2_rmse(y, m.fittedvalues)["rmse"],
            "rmse_leave_one_out": r2_rmse(y, held_out)["rmse"],
            "worst_leave_one_out_miss": float(np.max(np.abs(y.values - np.array(held_out)))),
            "coefficients": {k: {"coef": float(m.params[k]), "p_value": float(m.pvalues[k])} for k in m.params.index},
        }
    print("Candidate models (spreadsheet M1-M5 and v1):")
    print(pd.DataFrame({k: {c: v[c] for c in ("r2", "adj_r2", "rmse_all_days", "rmse_leave_one_out")}
                        for k, v in candidates.items()}).T.round(3).to_string(), "\n")

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
        "candidate_models": candidates,
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

    # Data file: the 8-row modelling table, one row per person per day.
    data_cols = {
        "Name": "who logged the day",
        "date": "the day (YYYY-MM-DD)",
        "fatigue_score": "OUTCOME: self-rated fatigue that day, 1 = fully charged, 10 = drained",
        "total_hours": "hours of logged interactions that day (v1's feature)",
        "interaction_count": "number of interactions logged that day",
        "social_day_share": "friend + family minutes / 1,440",
        "academic_day_share": "lecture + study + group work + teacher minutes / 1,440",
        "work_day_share": "work meeting minutes / 1,440",
        "weekend_indicator": "1 = Saturday or Sunday, 0 = weekday",
        "predicted_v1": "v1's predicted fatigue",
        "nudge_v1": "v1's decision: nudge if predicted fatigue >= 7",
    }
    t.assign(date=t["date"].dt.strftime("%Y-%m-%d"))[list(data_cols)].round(4).to_csv(
        HERE / "fatigue_model_data.csv", index=False)

    # Chart 2: typical error on all days vs on held-out days, per model.
    names = ["Baseline"] + list(candidates)
    in_sample = [fit["baseline_all_days"]["rmse"]] + [candidates[k]["rmse_all_days"] for k in candidates]
    held_out = [fit["baseline_leave_one_out"]["rmse"]] + [candidates[k]["rmse_leave_one_out"] for k in candidates]
    cap = 8
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = np.arange(len(names))[::-1]
    h = 0.36
    ax.barh(rows + h / 2, in_sample, height=h - 0.04, color=mark, label="all 8 days (the fit itself)")
    ax.barh(rows - h / 2, [min(v, cap) for v in held_out], height=h - 0.04, color=accent,
            label="held-out days (predicted from the other 7)")
    label_bg = {"facecolor": "white", "edgecolor": "none", "pad": 1}  # keeps the dashed line off the numbers
    for r, a, b in zip(rows, in_sample, held_out):
        ax.text(a + 0.08, r + h / 2, f"{a:.1f}", va="center", fontsize=8.5, color=muted, bbox=label_bg)
        ax.text(min(b, cap) + 0.08, r - h / 2, f"{b:.1f}" + ("  (off scale)" if b > cap else ""),
                va="center", fontsize=8.5, color=ink if b > cap else muted, weight="bold" if b > cap else "normal",
                bbox=label_bg)
    ax.axvline(fit["baseline_leave_one_out"]["rmse"], color=muted, lw=1.2, ls="--", zorder=0)
    ax.text(fit["baseline_leave_one_out"]["rmse"] + 0.08, len(names) - 0.35, "baseline on held-out days",
            fontsize=8.5, color=muted, va="bottom")
    ax.set_yticks(rows)
    ax.set_yticklabels([n.replace("v1 Total hours", "v1: total hours") for n in names], color=ink, fontsize=9.5)
    ax.set_xlim(0, cap + 2.2)
    ax.set_ylim(-0.7, len(names) - 0.1)
    ax.set_xlabel("Typical error in fatigue points (RMSE, lower is better)", color=muted)
    ax.set_title("Fits the 8 days vs predicts a new day", loc="left", color=ink, fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, left=False)
    fig.tight_layout()
    fig.savefig(HERE / "fatigue_model_comparison.png", dpi=200, facecolor="white")

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
    L += ["", "### Data dictionary (`fatigue_model_data.csv`)", "", "| Column | Meaning |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in data_cols.items()]
    L += ["", "## Choosing v1: the team spreadsheet's models vs total hours", "",
          "M1-M5 reproduce `daily_fatigue_regression.xlsx` (day share = minutes / 1,440; class discussion "
          "left out of the shares; weekend = Saturday/Sunday). Leave-one-out predicts each day from a model "
          "fit on the other 7, which shows how a model does on a day it has not seen.", "",
          "| Model | Features | Residual df | R² | Adjusted R² | Typical error, all days | Typical error, held-out days | Worst held-out miss |",
          "|---|---|---|---|---|---|---|---|",
          f"| Baseline | average fatigue | {len(t) - 1} | 0.000 | 0.000 | {fit['baseline_all_days']['rmse']:.2f} | "
          f"{fit['baseline_leave_one_out']['rmse']:.2f} | {max(abs(a - b) for a, b in zip(y, loo_base)):.1f} |"]
    for name, m in candidates.items():
        L.append(f"| {name} | {m['features'].replace(' + ', ', ')} | {m['residual_df']} | {m['r2']:.3f} | {m['adj_r2']:.3f} | "
                 f"{m['rmse_all_days']:.2f} | {m['rmse_leave_one_out']:.2f} | {m['worst_leave_one_out_miss']:.1f} |")
    best_sheet = max(SHEET_MODELS, key=lambda k: candidates[k]["adj_r2"])
    L += ["", f"{best_sheet} has the highest adjusted R² ({candidates[best_sheet]['adj_r2']:.3f}), but with "
              f"{candidates[best_sheet]['parameters']} numbers fit to {len(t)} days it misses held-out days by "
              f"{candidates[best_sheet]['rmse_leave_one_out']:.1f} points on average on a 1-10 scale. Every spreadsheet "
              "model does worse than the baseline on held-out days. Total hours is the only candidate that beats "
              "the baseline there, so it is v1.", "",
          "![model comparison](fatigue_model_comparison.png)", ""]
    L += ["## Model v1", "", f"`{FORMULA}` (OLS)", "",
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
          f"Value per user-day (ASSUMED): nudge on a tired day {usd(VALUE['hit'])}, nudge on a fine day "
          f"{usd(VALUE['false_alarm'])}, no nudge on a tired day {usd(VALUE['miss'])}, no nudge on a fine day "
          f"{usd(VALUE['quiet_ok'])}. Per year = per user-day x {USERS:,} users (ASSUMED) x {DAYS_PER_YEAR} days.", "",
          "| Rule | Nudges | Tired days caught | False alarms | Tired days missed | Value, 8 days | Per user-day | Per year |",
          "|---|---|---|---|---|---|---|---|"]
    for label, key in [("Baseline", "baseline"), ("v1", "v1"), ("v1, leave-one-out", "v1_leave_one_out")]:
        d = decisions[key]
        L.append(f"| {label} | {d['nudges']} | {d['hit']} | {d['false_alarm']} | {d['miss']} | {usd(d['value_total'])} | "
                 f"{usd(d['value_per_user_day'])} | {usd0(d['value_per_year'])} |")
    same = all((a >= TIRED_AT) == (b >= TIRED_AT) for a, b in zip(fitted, loo_v1))
    L += ["", "The leave-one-out row predicts each day from a model fit on the other 7. "
          + ("It makes the same nudge decision as v1 on every day, so the value is the same."
             if same else "Its nudge decisions differ from v1 on some days."), ""]
    L += ["## New input", "",
          f"{NEW_DAY['description']} = {new_hours:.1f} hours. v1 predicts fatigue {new_day['predicted_fatigue']:.2f}: "
          f"**{new_day['decision_v1']}**. Baseline: {new_day['decision_baseline']}.", ""]
    (HERE / "fatigue_model_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
