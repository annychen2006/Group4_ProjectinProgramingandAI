"""Used car prices: should the dealer buy this car at auction?

Follows "Regression assignment.md":
  1. Random 70/30 train/test split of data/used_car_sales.csv (240 cars sold
     in the last year).
  2. Fit OLS of Sale_Price on all five features (full model), then refit
     without Num_Owners, which does not earn its place (p = 0.87, adjusted R^2
     goes up without it). The refit is v1.
  3. Baseline (training-set average price) vs v1 on the test cars: R^2 and
     typical error (RMSE) in dollars, plus a predicted-vs-actual chart.
  4. Decision rule as a function (buy if auction price + $1,500 < predicted
     price), applied with the baseline rule to the 40 cars in
     data/auction_lots.csv. Profit per bought car = Later_Sale_Price -
     Auction_Price - 1,500.

Usage: python analysis/car_regression.py
Writes analysis/car_regression_report.md (every number),
analysis/car_regression_results.json, analysis/car_auction_decisions.csv
and analysis/car_pred_vs_actual.png.
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
SEED = 42
TEST_SHARE = 0.30
PREP_COST = 1_500  # inspection, reconditioning, detailing, lot time, per car

FULL = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners"
V1 = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)"
DROPPED = "Num_Owners"


def r2_rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    sse = float(((actual - predicted) ** 2).sum())
    sst = float(((actual - actual.mean()) ** 2).sum())
    return {"r2": 1 - sse / sst, "rmse": (sse / len(actual)) ** 0.5}


def coef_table(model):
    return {name: {"coef": float(model.params[name]), "p_value": float(model.pvalues[name])}
            for name in model.params.index}


def main():
    cars = pd.read_csv(DATA / "used_car_sales.csv")
    lots = pd.read_csv(DATA / "auction_lots.csv")

    # 1. Split: shuffle with a fixed seed, first 30% are the test cars.
    test = cars.sample(frac=TEST_SHARE, random_state=SEED)
    train = cars.drop(test.index)

    # 2. Fit.
    full = smf.ols(FULL, data=train).fit()
    v1 = smf.ols(V1, data=train).fit()
    print(full.summary())
    print(v1.summary())

    def predict_price(car):
        """v1's predicted lot sale price for one car (a dict or row of the five features)."""
        return float(v1.predict(pd.DataFrame([dict(car)]))[0])

    def should_buy(car, auction_price):
        """The dealer's rule: buy if auction price + prep cost is below v1's predicted price."""
        return "buy" if auction_price + PREP_COST < predict_price(car) else "pass"

    baseline_price = float(train["Sale_Price"].mean())

    def should_buy_baseline(auction_price):
        return "buy" if auction_price + PREP_COST < baseline_price else "pass"

    # 3. Fit on the test cars.
    test_pred_v1 = v1.predict(test)
    fit = {
        "baseline": r2_rmse(test["Sale_Price"], [baseline_price] * len(test)),
        "v1": r2_rmse(test["Sale_Price"], test_pred_v1),
        "v1_train": r2_rmse(train["Sale_Price"], v1.fittedvalues),
    }

    # 4. Decide on the 40 auction lots.
    lots["Predicted_Price_v1"] = v1.predict(lots).round(0)
    lots["Decision_v1"] = [should_buy(r, r["Auction_Price"]) for _, r in lots.iterrows()]
    lots["Decision_baseline"] = [should_buy_baseline(p) for p in lots["Auction_Price"]]
    lots["Profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
    lots.to_csv(HERE / "car_auction_decisions.csv", index=False)

    def outcome(col):
        bought = lots[lots[col] == "buy"]
        return {
            "cars_bought": int(len(bought)),
            "money_losing": int((bought["Profit_if_bought"] < 0).sum()),
            "total_profit": int(bought["Profit_if_bought"].sum()),
            "profit_per_car_bought": float(bought["Profit_if_bought"].mean()) if len(bought) else 0.0,
            "lots_bought": bought["Lot"].tolist(),
        }

    auction = {
        "baseline": outcome("Decision_baseline"),
        "v1": outcome("Decision_v1"),
        "buy_every_car": {
            "cars_bought": len(lots),
            "money_losing": int((lots["Profit_if_bought"] < 0).sum()),
            "total_profit": int(lots["Profit_if_bought"].sum()),
        },
    }

    results = {
        "n_cars": len(cars), "n_train": len(train), "n_test": len(test), "seed": SEED,
        "n_auction_lots": len(lots), "prep_cost": PREP_COST,
        "accident_yes": int((cars["Accident"] == "yes").sum()),
        "full_model": {"formula": FULL, "r2": float(full.rsquared), "adj_r2": float(full.rsquared_adj),
                       "coefficients": coef_table(full)},
        "v1": {"formula": V1, "dropped": DROPPED, "r2": float(v1.rsquared), "adj_r2": float(v1.rsquared_adj),
               "residual_std_error": float(np.sqrt(v1.mse_resid)), "coefficients": coef_table(v1)},
        "baseline_price": baseline_price,
        "fit": fit,
        "auction": auction,
    }
    (HERE / "car_regression_results.json").write_text(json.dumps(results, indent=2))

    # Chart: predicted vs actual on the test cars, baseline and v1 side by side.
    ink, muted, grid, mark = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    lo, hi = 0, 50_000
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, (title, pred, stats) in zip(axes, [
        ("Baseline: same price for every car", [baseline_price] * len(test), fit["baseline"]),
        ("v1: regression", test_pred_v1, fit["v1"]),
    ]):
        ax.plot([lo, hi], [lo, hi], color=muted, lw=1.2, ls="--", zorder=1)
        ax.text(hi * 0.05, hi * 0.07, "perfect prediction", color=muted, fontsize=9,
                ha="left", rotation=45, rotation_mode="anchor")
        ax.scatter(test["Sale_Price"], pred, s=34, color=mark, edgecolor="white", linewidth=1, zorder=2)
        ax.set_title(title, loc="left", fontsize=12, color=ink, pad=22)
        ax.text(0, 1.02, f"R² {stats['r2']:.2f}   typical error ${stats['rmse']:,.0f}",
                transform=ax.transAxes, fontsize=10, color=muted)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.xaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
        ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
        ax.set_xlabel("Actual sale price", color=muted, fontsize=10)
    axes[0].set_ylabel("Predicted sale price", color=muted, fontsize=10)
    fig.suptitle(f"Predicted vs actual sale price, {len(test)} test cars the model never saw",
                 x=0.01, ha="left", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(HERE / "car_pred_vs_actual.png", dpi=200, facecolor="white")

    # Report.
    usd = lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    pfmt = lambda p: "< 0.0001" if p < 0.0001 else f"{p:.4f}"
    L = ["# Used car prices: regression results", "",
         "## 1. Data and split", "",
         f"- `used_car_sales.csv`: {len(cars)} cars the dealer sold in the last year, "
         f"{results['accident_yes']} with an accident on record.",
         f"- Random 70/30 split (seed {SEED}): {len(train)} training cars, {len(test)} test cars.",
         f"- `auction_lots.csv`: {len(lots)} cars from one recent auction, used only to test the buy rule.", "",
         "## 2. Model", ""]
    for label, m, key in [("Full model", full, "full_model"), ("v1 (Num_Owners dropped)", v1, "v1")]:
        L += [f"**{label}**: `{m.model.formula}`", "",
              "| Term | Coefficient | p-value |", "|---|---|---|"]
        for name, c in results[key]["coefficients"].items():
            L.append(f"| {name} | {c['coef']:,.1f} | {pfmt(c['p_value'])} |")
        L += ["", f"R² {m.rsquared:.4f}, adjusted R² {m.rsquared_adj:.4f}.", ""]
    L += [f"Num_Owners has p = {full.pvalues['Num_Owners']:.2f} in the full model, and dropping it raises "
          f"adjusted R² from {full.rsquared_adj:.4f} to {v1.rsquared_adj:.4f}. v1 is the model without it.", "",
          "## 3. Fit on the test cars", "",
          "| Model | R² | Typical error (RMSE) |", "|---|---|---|",
          f"| Baseline (training average, {usd(baseline_price)}) | {fit['baseline']['r2']:.3f} | {usd(fit['baseline']['rmse'])} |",
          f"| v1 | {fit['v1']['r2']:.3f} | {usd(fit['v1']['rmse'])} |",
          f"| v1 on its own training cars | {fit['v1_train']['r2']:.3f} | {usd(fit['v1_train']['rmse'])} |", "",
          "![predicted vs actual](car_pred_vs_actual.png)", "",
          "## 4. Decisions on the 40 auction lots", "",
          f"Rule: buy if auction price + ${PREP_COST:,} < predicted price, else pass "
          "(`should_buy` in car_regression.py). Profit per bought car = Later_Sale_Price - Auction_Price - 1,500.", "",
          "| Rule | Cars bought | Money-losing cars bought | Total profit | Profit per car bought |",
          "|---|---|---|---|---|"]
    for label, key in [("Baseline", "baseline"), ("v1", "v1")]:
        a = auction[key]
        L.append(f"| {label} | {a['cars_bought']} | {a['money_losing']} | {usd(a['total_profit'])} | "
                 f"{usd(a['profit_per_car_bought'])} |")
    b = auction["buy_every_car"]
    L += [f"| (for reference) buy all {len(lots)} | {b['cars_bought']} | {b['money_losing']} | {usd(b['total_profit'])} | |",
          "", "Per-lot predictions and decisions: `car_auction_decisions.csv`.", ""]
    (HERE / "car_regression_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
