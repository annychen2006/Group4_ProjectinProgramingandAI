"""Social Battery v1: should the app tell this user to recharge today?

Group model assignment (regression). Loads the team's logs, fits v1, prints
the fit evaluation, values the nudge rule against the baseline, and returns
a decision for a new day.

Data (self-logged by the four team members, 14-19 October 2026):
  data/interactions.csv   one row per interaction (type, minutes, who, date)
  data/daily_fatigue.csv  one row per person per day, with the outcome
                          fatigue_score (1 = fully charged, 10 = drained)

Model v1 = M5, the full model from the team spreadsheet
(daily_fatigue_regression.xlsx): OLS of fatigue_score on interaction_count,
social/academic/work day shares (minutes / 1,440) and weekend_indicator.
8 person-days is under 30, so per the brief there is no 70/30 split: v1 is
fit on all 8 rows. As a check on new days, each day is also predicted by a
model fit on the other 7 (leave-one-out).

Decision rule: nudge ("your battery is low, block some recharge time") if
predicted fatigue >= TIRED_AT, otherwise stay quiet.
Baseline: predict the average fatigue for every day, so the same decision
every day.

Usage: python analysis/fatigue_model.py
Writes analysis/fatigue_model_report.md, analysis/fatigue_model_results.json,
analysis/fatigue_model_data.csv, analysis/fatigue_model_chart.png and
analysis/fatigue_model_comparison.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MINUTES_PER_DAY = 1440

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

# One realistic new day the app might see.
NEW_DAY = {
    "description": "weekday: 2 h lecture, 1.5 h study, 1 h group work, 1.5 h with friends",
    "weekend": False,
    "interactions": [("lecture", 120), ("study", 90), ("group work", 60), ("friend", 90)],
}

# The team spreadsheet's definitions: day share = minutes / 1,440; "class
# discussion" is not in the category mapping, so its minutes are left out of
# the shares (it still counts as an interaction); weekend = Saturday/Sunday.
CATEGORIES = {
    "group work": "academic", "study": "academic", "lecture": "academic", "teacher": "academic",
    "work meeting": "work", "friend": "social", "family": "social",
}
FEATURES = ["interaction_count", "social_day_share", "academic_day_share", "work_day_share", "weekend_indicator"]
SHEET_MODELS = {
    "M1 Count only": "interaction_count",
    "M2 Time shares only": "social_day_share + academic_day_share + work_day_share",
    "M3 Calendar only": "weekend_indicator",
    "M4 Interaction features": "interaction_count + social_day_share + academic_day_share + work_day_share",
    "M5 Full model": " + ".join(FEATURES),
}
V1_NAME = "M5 Full model"
FORMULA = f"fatigue_score ~ {SHEET_MODELS[V1_NAME]}"
ALTERNATIVE = {"Total hours (alternative)": "total_hours"}


def day_features(interactions, weekend):
    """The five model features for one day, from a list of (type, minutes)."""
    minutes = {"social": 0, "academic": 0, "work": 0}
    for kind, mins in interactions:
        category = CATEGORIES.get(kind.strip().lower())
        if category:
            minutes[category] += mins
    return {
        "interaction_count": len(interactions),
        "social_day_share": minutes["social"] / MINUTES_PER_DAY,
        "academic_day_share": minutes["academic"] / MINUTES_PER_DAY,
        "work_day_share": minutes["work"] / MINUTES_PER_DAY,
        "weekend_indicator": int(weekend),
    }


def load():
    """One row per person per day: the outcome plus the features, built from the raw logs."""
    interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["date"])
    fatigue = pd.read_csv(DATA / "daily_fatigue.csv", parse_dates=["date"])
    rows = []
    for (name, date), day in interactions.groupby(["Name", "date"]):
        feats = day_features(list(zip(day["type"], day["duration_min"])), date.dayofweek >= 5)
        rows.append({"Name": name, "date": date, "total_hours": day["duration_min"].sum() / 60, **feats})
    table = fatigue[["Name", "date", "fatigue_score"]].merge(pd.DataFrame(rows), on=["Name", "date"], how="left")
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
    print(t.assign(date=t["date"].dt.strftime("%a %d %b")).round(3).to_string(index=False), "\n")

    v1 = smf.ols(FORMULA, data=t).fit()
    print(v1.summary(), "\n")
    average = float(y.mean())

    def predict_fatigue(features):
        """v1's predicted fatigue score for one day's features."""
        return float(v1.predict(pd.DataFrame([features]))[0])

    def should_nudge(day_interactions, weekend):
        """The app's rule for one day: (list of (type, minutes), weekend?) -> 'nudge' or 'stay quiet'."""
        return "nudge" if predict_fatigue(day_features(day_interactions, weekend)) >= TIRED_AT else "stay quiet"

    fitted = v1.fittedvalues
    loo_v1 = leave_one_out(FORMULA, t)
    loo_base = [float(t.drop(i)["fatigue_score"].mean()) for i in t.index]
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

    # Every candidate: the spreadsheet's M1-M5 plus total hours, on all days and on held-out days.
    candidates = {}
    for name, rhs in {**SHEET_MODELS, **ALTERNATIVE}.items():
        formula = f"fatigue_score ~ {rhs}"
        m = smf.ols(formula, data=t).fit()
        held_out = leave_one_out(formula, t)
        candidates[name] = {
            "features": rhs, "parameters": int(len(m.params)), "residual_df": int(m.df_resid),
            "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "f_test_p": float(m.f_pvalue),
            "rmse_all_days": r2_rmse(y, m.fittedvalues)["rmse"],
            "rmse_leave_one_out": r2_rmse(y, held_out)["rmse"],
            "worst_leave_one_out_miss": float(np.max(np.abs(y.values - np.array(held_out)))),
            "decisions_leave_one_out": outcome(y, held_out),
            "coefficients": {k: {"coef": float(m.params[k]), "p_value": float(m.pvalues[k])} for k in m.params.index},
        }
    print("Candidate models:")
    print(pd.DataFrame({k: {c: v[c] for c in ("r2", "adj_r2", "rmse_all_days", "rmse_leave_one_out")}
                        for k, v in candidates.items()}).T.round(3).to_string(), "\n")

    new_features = day_features(NEW_DAY["interactions"], NEW_DAY["weekend"])
    new_day = {**NEW_DAY, "features": new_features,
               "predicted_fatigue": predict_fatigue(new_features),
               "decision_v1": should_nudge(NEW_DAY["interactions"], NEW_DAY["weekend"]),
               "decision_baseline": "nudge" if average >= TIRED_AT else "stay quiet"}

    t["predicted_v1"] = fitted
    t["predicted_v1_held_out"] = loo_v1
    t["nudge_v1"] = np.where(fitted >= TIRED_AT, "nudge", "stay quiet")
    coefs = {k: {"coef": float(v1.params[k]), "p_value": float(v1.pvalues[k])} for k in v1.params.index}
    results = {
        "rows": len(t), "people": sorted(t["Name"].unique().tolist()),
        "period": [t["date"].min().strftime("%Y-%m-%d"), t["date"].max().strftime("%Y-%m-%d")],
        "interactions_logged": int(len(pd.read_csv(DATA / "interactions.csv"))),
        "split": "none: 8 rows < 30, v1 fit on all rows (leave-one-out used as a held-out check)",
        "v1": V1_NAME, "formula": FORMULA, "coefficients": coefs,
        "r2": float(v1.rsquared), "adj_r2": float(v1.rsquared_adj), "f_test_p": float(v1.f_pvalue),
        "residual_df": int(v1.df_resid),
        "per_hour": {f: coefs[f]["coef"] / 24 for f in ("social_day_share", "academic_day_share", "work_day_share")},
        "average_fatigue": average, "tired_at": TIRED_AT, "tired_days": int((y >= TIRED_AT).sum()),
        "candidate_models": candidates,
        "fit": fit, "value_assumptions": VALUE, "users_assumed": USERS,
        "decisions": decisions, "new_day": new_day,
        "per_day": t.assign(date=t["date"].dt.strftime("%Y-%m-%d")).round(3).to_dict("records"),
    }
    (HERE / "fatigue_model_results.json").write_text(json.dumps(results, indent=2))

    # Data file: the 8-row modelling table, one row per person per day.
    data_cols = {
        "Name": "who logged the day",
        "date": "the day (YYYY-MM-DD)",
        "fatigue_score": "OUTCOME: self-rated fatigue that day, 1 = fully charged, 10 = drained",
        "interaction_count": "number of interactions logged that day",
        "social_day_share": "friend + family minutes / 1,440",
        "academic_day_share": "lecture + study + group work + teacher minutes / 1,440 (class discussion left out)",
        "work_day_share": "work meeting minutes / 1,440",
        "weekend_indicator": "1 = Saturday or Sunday, 0 = weekday",
        "total_hours": "hours of logged interactions that day (used only by the alternative model)",
        "predicted_v1": "v1's predicted fatigue, fit on all 8 days",
        "predicted_v1_held_out": "v1's predicted fatigue when this day is left out of the fit",
        "nudge_v1": "v1's decision: nudge if predicted fatigue >= 7",
    }
    t.assign(date=t["date"].dt.strftime("%Y-%m-%d"))[list(data_cols)].round(4).to_csv(
        HERE / "fatigue_model_data.csv", index=False)

    ink, muted, grid, mark, accent = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"

    def style(ax):
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)

    # Chart 1: predicted vs actual fatigue, baseline vs v1.
    lo, hi = 0, 12
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    panels = [
        ("Baseline: average for every day", [average] * len(y), None, fit["baseline_all_days"]),
        (f"v1 ({V1_NAME.split()[0]})", fitted, loo_v1, fit["v1_all_days"]),
    ]
    for ax, (title, pred, held, stats) in zip(axes, panels):
        ax.axhspan(TIRED_AT, hi, color="#f4f3f0", zorder=0)
        ax.text(0.3, hi - 0.3, f"predicted {TIRED_AT}+: app nudges", color=muted, fontsize=9, va="top")
        ax.plot([lo, 10], [lo, 10], color=muted, lw=1.2, ls="--", zorder=1)
        ax.text(0.45, 0.75, "perfect prediction", color=muted, fontsize=8.5, ha="left", rotation=45,
                rotation_mode="anchor")
        ax.scatter(y, pred, s=46, color=mark, edgecolor="white", linewidth=1, zorder=3,
                   label="fit on all 8 days")
        if held is not None:
            shown = np.clip(held, lo + 0.2, hi - 0.2)
            ax.scatter(y, shown, s=46, facecolor="white", edgecolor=accent, linewidth=1.6, zorder=2,
                       label="day left out of the fit")
            for actual, h, s in zip(y, held, shown):
                if h != s:
                    high = h > s
                    ax.annotate(f"{h:.0f}" if abs(h) >= 10 else f"{h:.1f}", (actual, s), textcoords="offset points",
                                xytext=(-8, -4) if high else (8, -3), ha="right" if high else "left",
                                fontsize=8.5, color=ink, weight="bold")
                    ax.annotate("off scale", (actual, s), textcoords="offset points",
                                xytext=(-8, -15) if high else (8, 9), ha="right" if high else "left",
                                fontsize=8, color=muted)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, color=ink, pad=22)
        ax.text(0, 1.02, f"R² {stats['r2']:.2f}   typical error {stats['rmse']:.1f} points (all 8 days)",
                transform=ax.transAxes, fontsize=10, color=muted)
        ax.set_xlim(lo, 10.5)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual fatigue (1-10)", color=muted)
        style(ax)
    axes[0].set_ylabel("Predicted fatigue", color=muted)
    fig.suptitle(f"Predicted vs actual fatigue, {len(t)} logged days", x=0.01, ha="left", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(HERE / "fatigue_model_chart.png", dpi=200, facecolor="white")
    plt.close(fig)

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
    ax.set_yticklabels([f"{n}  (v1)" if n == V1_NAME else n for n in names], color=ink, fontsize=9.5)
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
    plt.close(fig)

    # Report.
    usd = lambda v: f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    usd0 = lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    ph = results["per_hour"]
    L = ["# Social Battery v1: fatigue model results", "",
         "## Data", "",
         f"- Self-logged by the team ({', '.join(results['people'])}), {results['period'][0]} to {results['period'][1]}.",
         f"- `interactions.csv`: {results['interactions_logged']} interactions. `daily_fatigue.csv`: {len(t)} person-days (the outcome).",
         f"- {len(t)} rows is under 30, so no 70/30 split: v1 is fit on all {len(t)} rows. "
         "Leave-one-out (predict each day from the other 7) is reported as a check on new days.", "",
         "| Person | Date | Interactions | Social share | Academic share | Work share | Weekend | Fatigue | v1 predicted | v1 predicted, day left out | v1 decision |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results["per_day"]:
        L.append(f"| {r['Name']} | {r['date']} | {r['interaction_count']} | {r['social_day_share']:.3f} | "
                 f"{r['academic_day_share']:.3f} | {r['work_day_share']:.3f} | {r['weekend_indicator']} | "
                 f"{r['fatigue_score']} | {r['predicted_v1']:.2f} | {r['predicted_v1_held_out']:.2f} | {r['nudge_v1']} |")
    L += ["", "### Data dictionary (`fatigue_model_data.csv`)", "", "| Column | Meaning |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in data_cols.items()]
    L += ["", f"## Model v1: {V1_NAME} from the team spreadsheet", "", f"`{FORMULA}` (OLS)", "",
          "| Term | Coefficient | p-value | In plain words (other features held equal) |", "|---|---|---|---|"]
    plain = {
        "Intercept": "starting point",
        "interaction_count": f"each extra interaction: {sign(coefs['interaction_count']['coef'])} points",
        "social_day_share": f"each hour with friends/family: {sign(ph['social_day_share'])} points",
        "academic_day_share": f"each hour of class/study: {sign(ph['academic_day_share'])} points",
        "work_day_share": f"each hour of work meetings: {sign(ph['work_day_share'])} points",
        "weekend_indicator": f"a weekend day: {sign(coefs['weekend_indicator']['coef'])} points",
    }
    for k, v in coefs.items():
        L.append(f"| {k} | {v['coef']:.3f} | {v['p_value']:.3f} | {plain[k]} |")
    L += ["", f"R² {v1.rsquared:.3f}, adjusted R² {v1.rsquared_adj:.3f} (F-test p = {v1.f_pvalue:.3f}), "
              f"with {int(v1.df_resid)} residual degrees of freedom: {len(v1.params)} numbers fit to {len(t)} days. "
              "A day share is minutes / 1,440, so a share coefficient / 24 is the effect of one hour.", "",
          "## Fit, baseline vs v1", "",
          "| | R² | Typical error (RMSE, fatigue points) |", "|---|---|---|",
          f"| Baseline (average {average:.2f}), all {len(t)} days | {fit['baseline_all_days']['r2']:.3f} | {fit['baseline_all_days']['rmse']:.2f} |",
          f"| v1, all {len(t)} days | {fit['v1_all_days']['r2']:.3f} | {fit['v1_all_days']['rmse']:.2f} |",
          f"| Baseline, leave-one-out | {fit['baseline_leave_one_out']['r2']:.3f} | {fit['baseline_leave_one_out']['rmse']:.2f} |",
          f"| v1, leave-one-out | {fit['v1_leave_one_out']['r2']:.3f} | {fit['v1_leave_one_out']['rmse']:.2f} |", "",
          "![predicted vs actual](fatigue_model_chart.png)", "",
          "On all 8 days v1 is nearly exact, but that is expected with 6 numbers fit to 8 days. With a day left out, "
          f"its predictions swing far off the 1-10 scale (worst: {candidates[V1_NAME]['worst_leave_one_out_miss']:.0f} "
          "points off), so its predicted numbers should not be shown to users. Its nudge decisions hold up better "
          "(see below), because a decision only depends on which side of 7 the prediction falls.", "",
          "### All candidate models", "",
          "M1-M5 reproduce `daily_fatigue_regression.xlsx`; total hours is a one-feature alternative.", "",
          "| Model | Residual df | R² | Adjusted R² | Typical error, all days | Typical error, held-out days | Tired days caught, held-out | False alarms, held-out |",
          "|---|---|---|---|---|---|---|---|"]
    for name, m in candidates.items():
        d = m["decisions_leave_one_out"]
        L.append(f"| {name}{' (v1)' if name == V1_NAME else ''} | {m['residual_df']} | {m['r2']:.3f} | {m['adj_r2']:.3f} | "
                 f"{m['rmse_all_days']:.2f} | {m['rmse_leave_one_out']:.2f} | {d['hit']} of {results['tired_days']} | {d['false_alarm']} |")
    L += ["", "![model comparison](fatigue_model_comparison.png)", "",
          "## Decision and value", "",
          f"Rule: nudge if predicted fatigue >= {TIRED_AT}, else stay quiet (`should_nudge`). "
          f"{results['tired_days']} of {len(t)} logged days were tired days.", "",
          f"Value per user-day (ASSUMED): nudge on a tired day {usd(VALUE['hit'])}, nudge on a fine day "
          f"{usd(VALUE['false_alarm'])}, no nudge on a tired day {usd(VALUE['miss'])}, no nudge on a fine day "
          f"{usd(VALUE['quiet_ok'])}. Per year = per user-day x {USERS:,} users (ASSUMED) x {DAYS_PER_YEAR} days.", "",
          "| Rule | Nudges | Tired days caught | False alarms | Tired days missed | Value, 8 days | Per user-day | Per year |",
          "|---|---|---|---|---|---|---|---|"]
    for label, key in [("Baseline (never nudge)", "baseline"), ("v1, fit on all 8 days", "v1"),
                       ("v1, each day left out", "v1_leave_one_out")]:
        d = decisions[key]
        L.append(f"| {label} | {d['nudges']} | {d['hit']} | {d['false_alarm']} | {d['miss']} | {usd(d['value_total'])} | "
                 f"{usd(d['value_per_user_day'])} | {usd0(d['value_per_year'])} |")
    gain_fit = decisions["v1"]["value_per_year"] - decisions["baseline"]["value_per_year"]
    gain_loo = decisions["v1_leave_one_out"]["value_per_year"] - decisions["baseline"]["value_per_year"]
    L += ["", f"v1 vs baseline per year: {usd0(gain_fit)} better on the days it was fit on, {usd0(gain_loo)} better "
              "when each day is predicted without seeing it. The second is the more honest figure.", "",
          "## New input", "",
          f"{NEW_DAY['description']}: {new_features['interaction_count']} interactions, social share "
          f"{new_features['social_day_share']:.3f}, academic share {new_features['academic_day_share']:.3f}, "
          f"work share {new_features['work_day_share']:.3f}, weekend {new_features['weekend_indicator']}. "
          f"v1 predicts fatigue {new_day['predicted_fatigue']:.1f}: **{new_day['decision_v1']}**. "
          f"Baseline: {new_day['decision_baseline']}."]
    if not 1 <= new_day["predicted_fatigue"] <= 10:
        L.append("The predicted score is outside the 1-10 scale, another sign that v1's numbers are unstable; "
                 "the decision only uses whether it is 7 or more.")
    L.append("")
    (HERE / "fatigue_model_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
