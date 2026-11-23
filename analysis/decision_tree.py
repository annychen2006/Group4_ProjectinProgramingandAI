"""Clinic no-shows: should the clinic overbook this slot?

Follows "Decision tree assignment.md":
  1. Stratified 70/30 train/test split of data/clinic_appointments.csv.
  2. v1: one classification tree fit on the training appointments.
  3. Baseline (predict show for everyone) vs v1 on the test appointments:
     confusion matrix, accuracy, precision, recall; v1 train vs test accuracy.
  4. Decision rule as a function, applied to appointments D and E.
  5. Dollar value from the assignment's cost table, total / per appointment /
     per year at an assumed volume.

Usage: python analysis/decision_tree.py
Writes analysis/decision_tree_results.json, analysis/decision_tree_report.md
and analysis/decision_tree.png (the tree picture).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "clinic_appointments.csv"
SEED = 42
TEST_SIZE = 0.30
MAX_DEPTH = 3      # see the cross-validation table in the report for why
MIN_LEAF = 10      # no leaf smaller than 10 training appointments
FEATURES = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]

# Cost table from the assignment, dollars per appointment.
# Positive class = no-show; predicting no-show means overbooking the slot.
VALUE = {
    "TN": 0,     # hold, patient came
    "FP": -60,   # overbook, patient came: overtime and copay refund
    "FN": -120,  # hold, patient did not come: slot sat empty
    "TP": 0,     # overbook, patient did not come: second patient filled the slot
}
APPOINTMENTS_PER_YEAR = 10_000  # ASSUMPTION (about 40 a day, 250 days). Change to the clinic's volume.

UPCOMING = {
    "D": {"lead_time_days": 21, "prior_no_shows": 1, "age": 27, "distance_miles": 9.5,
          "sms_reminder": 0, "new_patient": 1, "appointment_hour": 15},
    "E": {"lead_time_days": 3, "prior_no_shows": 0, "age": 61, "distance_miles": 2.0,
          "sms_reminder": 1, "new_patient": 0, "appointment_hour": 9},
}


def cells(y, pred):
    y, pred = [int(v) for v in y], [int(v) for v in pred]
    tp = sum(a == 1 and p == 1 for a, p in zip(y, pred))
    fp = sum(a == 0 and p == 1 for a, p in zip(y, pred))
    fn = sum(a == 1 and p == 0 for a, p in zip(y, pred))
    tn = sum(a == 0 and p == 0 for a, p in zip(y, pred))
    n = len(y)
    value = tn * VALUE["TN"] + fp * VALUE["FP"] + fn * VALUE["FN"] + tp * VALUE["TP"]
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n": n,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "value_total": value,
        "value_per_appointment": value / n,
        "value_per_year": value / n * APPOINTMENTS_PER_YEAR,
    }


def decide(tree, appointment):
    """The decision rule: overbook if v1 predicts a no-show, otherwise hold."""
    row = pd.DataFrame([appointment])[FEATURES]
    return "overbook" if tree.predict(row)[0] == 1 else "hold"


def leaf_path(tree, appointment):
    """The questions this appointment answers on its way to a leaf, plus the leaf's training counts."""
    row = pd.DataFrame([appointment])[FEATURES]
    t = tree.tree_
    steps = []
    for node in tree.decision_path(row).indices:
        if t.children_left[node] == -1:
            leaf = int(node)
            continue
        name, cut = FEATURES[t.feature[node]], float(t.threshold[node])
        went_left = appointment[name] <= cut
        steps.append(f"{name} {'<=' if went_left else '>'} {cut:g}")
    n = int(t.n_node_samples[leaf])
    no_show = int(round(t.value[leaf][0][1] * n)) if t.value[leaf][0].sum() <= 1.0001 else int(t.value[leaf][0][1])
    return {"leaf_id": leaf, "path": steps, "train_n": n, "train_no_shows": no_show,
            "train_no_show_rate": no_show / n}


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = (df["no_show"] == "no-show").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # Depth check, on the training appointments only (test rows are never used to choose).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc = {
        d: float(cross_val_score(
            DecisionTreeClassifier(max_depth=d, min_samples_leaf=MIN_LEAF, random_state=SEED),
            X_tr, y_tr, cv=cv, scoring="accuracy").mean())
        for d in range(1, 7)
    }

    v1 = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED).fit(X_tr, y_tr)

    results = {
        "n": len(df), "n_train": len(X_tr), "n_test": len(X_te),
        "no_shows": int(y.sum()), "no_show_rate": float(y.mean()),
        "no_shows_train": int(y_tr.sum()), "no_show_rate_train": float(y_tr.mean()),
        "no_shows_test": int(y_te.sum()), "no_show_rate_test": float(y_te.mean()),
        "seed": SEED, "features_offered": FEATURES, "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_LEAF, "leaves": int(v1.get_n_leaves()),
        "cv_accuracy_by_depth": {str(d): a for d, a in cv_acc.items()},
        "root_split": f"{FEATURES[v1.tree_.feature[0]]} <= {float(v1.tree_.threshold[0]):g}",
        "features_split_on": {f: float(i) for f, i in
                              sorted(zip(FEATURES, v1.feature_importances_), key=lambda kv: -kv[1]) if i > 0},
        "tree_rules": export_text(v1, feature_names=FEATURES, show_weights=True),
        "value_table": VALUE, "appointments_per_year_assumed": APPOINTMENTS_PER_YEAR,
        "baseline": {"train": cells(y_tr, [0] * len(y_tr)), "test": cells(y_te, [0] * len(y_te))},
        "v1": {"train": cells(y_tr, v1.predict(X_tr)), "test": cells(y_te, v1.predict(X_te))},
        "upcoming": {k: {"appointment": a, "decision": decide(v1, a), **leaf_path(v1, a)}
                     for k, a in UPCOMING.items()},
    }
    b, m = results["baseline"]["test"], results["v1"]["test"]
    results["v1_minus_baseline_test"] = {
        "value_total": m["value_total"] - b["value_total"],
        "value_per_appointment": m["value_per_appointment"] - b["value_per_appointment"],
        "value_per_year": m["value_per_year"] - b["value_per_year"],
    }
    (HERE / "decision_tree_results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(15, 6.5))
    plot_tree(v1, feature_names=FEATURES, class_names=["show", "no-show"], filled=True,
              rounded=True, impurity=False, fontsize=10, ax=ax)
    fig.savefig(HERE / "decision_tree.png", dpi=200, bbox_inches="tight", facecolor="white")

    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    usd = lambda v: f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
    L = ["# Clinic no-shows: decision tree results", "",
         "## 1. Data and split", "",
         f"- {results['n']} appointments, {results['no_shows']} no-shows ({pct(results['no_show_rate'])}).",
         f"- Train: {results['n_train']} appointments, {results['no_shows_train']} no-shows ({pct(results['no_show_rate_train'])}).",
         f"- Test: {results['n_test']} appointments, {results['no_shows_test']} no-shows ({pct(results['no_show_rate_test'])}).",
         f"- Stratified random split, seed {SEED}.", "",
         "## 2. Model v1", "",
         f"- Classification tree (CART, Gini), all {len(FEATURES)} features offered, max depth {MAX_DEPTH}, "
         f"at least {MIN_LEAF} training appointments per leaf, {results['leaves']} leaves.",
         f"- Root split: `{results['root_split']}`.",
         "- Features the tree splits on (share of total impurity reduction): "
         + ", ".join(f"{f} {pct(i)}" for f, i in results["features_split_on"].items()) + ".",
         "- Depth check, 5-fold cross-validated accuracy on the training appointments: "
         + ", ".join(f"depth {d}: {pct(a)}" for d, a in cv_acc.items()) + ".", "",
         "```", results["tree_rules"].rstrip(), "```",
         "(weights are [show, no-show] training appointments in each leaf)", "",
         "![tree](decision_tree.png)", "",
         "## 3. Fit on the test appointments", "",
         "| Model | TN | FP | FN | TP | Accuracy | Precision | Recall |", "|---|---|---|---|---|---|---|---|"]
    for name in ("baseline", "v1"):
        c = results[name]["test"]
        L.append(f"| {name} | {c['TN']} | {c['FP']} | {c['FN']} | {c['TP']} | {pct(c['accuracy'])} | "
                 f"{pct(c['precision'])} | {pct(c['recall'])} |")
    L += ["", f"v1 training accuracy {pct(results['v1']['train']['accuracy'])} vs test accuracy "
              f"{pct(m['accuracy'])}.", "",
          "## 4. Decision for D and E", "",
          "Rule: overbook if v1 predicts a no-show, hold otherwise (`decide()` in decision_tree.py).", ""]
    for k, u in results["upcoming"].items():
        L.append(f"- **{k}**: {' -> '.join(u['path'])} -> leaf {u['leaf_id']} "
                 f"({u['train_no_shows']} of {u['train_n']} training appointments were no-shows, "
                 f"{pct(u['train_no_show_rate'])}). Decision: **{u['decision']}**.")
    L += ["", "## 5. Value on the test appointments", "",
          f"Cost table: {VALUE}. Per-year figure assumes {APPOINTMENTS_PER_YEAR:,} appointments a year (assumption).", "",
          "| Model | Total | Per appointment | Per year |", "|---|---|---|---|"]
    for name in ("baseline", "v1"):
        c = results[name]["test"]
        L.append(f"| {name} | {usd(c['value_total'])} | {usd(c['value_per_appointment'])} | {usd(c['value_per_year'])} |")
    d = results["v1_minus_baseline_test"]
    L += [f"| v1 minus baseline | {usd(d['value_total'])} | {usd(d['value_per_appointment'])} | {usd(d['value_per_year'])} |", ""]
    (HERE / "decision_tree_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
