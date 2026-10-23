"""Explain the frozen logistic-regression PPH model.

For a logistic regression the explanation is exact (no approximation): the log-odds of
PPH = baseline + sum of one contribution per risk factor, where

    contribution_j = coefficient_j * (patient value_j - average training value_j)

and baseline is the log-odds of the "average" training patient. This is identical to
SHAP values for a linear model. The one-hot education columns are summed into a single
`education_level` contribution so the output is one number per clinical factor.

Provides:
  load(variant)                 -> bundle (load once, e.g. at backend startup)
  explain_patient(bundle, dict) -> JSON-friendly dict (risk, band, contributions)
  global_importance(bundle, df) -> mean |contribution| per factor

Run from the project root:  python src/clinical/explain.py
Saves: outputs/plots/global_importance.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from calibrate import HOLDOUT_FRACTION, MODEL_DIR, PLOT_DIR
from train import DATA_PATH, SEED, TARGET

LABELS = {
    "age": "Maternal age", "education_level": "Education level",
    "num_pregnancies": "Number of pregnancies", "parity": "Parity (previous births)",
    "prenatal_checkups": "Prenatal checkups", "prev_cesarean_count": "Previous C-sections",
    "cesarean_history": "History of C-section", "mode_of_delivery": "Vaginal delivery",
    "gestational_age": "Gestational age", "multiple_births": "Multiple births",
    "hypertensive_disorders": "Hypertensive disorders", "cardiac_disease": "Cardiac disease",
    "liver_disease": "Liver disease", "gestational_diabetes": "Gestational diabetes",
    "macrosomia": "Macrosomia (birth weight >= 4 kg)",
}


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _group_of(col_name: str) -> str:
    """'cat__education_level_3' -> 'education_level'; 'num__age' -> 'age'."""
    name = col_name.split("__", 1)[1]
    return "education_level" if name.startswith("education_level_") else name


def load(variant: str = "full") -> dict:
    record = json.loads((MODEL_DIR / "risk_bands.json").read_text())[variant]
    model = joblib.load(MODEL_DIR / record["model_file"])
    if not hasattr(model, "named_steps"):
        raise ValueError("explain.py needs the plain logistic-regression pipeline "
                         f"(calibration was '{record['calibration']}').")
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    features = record["features"]

    # reference = average patient of the TRAINING part (same split as calibrate.py).
    # Cached in models/explain_reference_<variant>.json so the app does not need data/.
    ref_path = MODEL_DIR / f"explain_reference_{variant}.json"
    if ref_path.exists():
        ref_mean = np.array(json.loads(ref_path.read_text())["ref_mean"])
    else:
        df = pd.read_csv(DATA_PATH)
        train_df, _ = train_test_split(df, test_size=HOLDOUT_FRACTION,
                                       stratify=df[TARGET], random_state=SEED)
        ref_mean = _dense(prep.transform(train_df[features])).mean(axis=0)
        ref_path.write_text(json.dumps({"variant": variant, "ref_mean": ref_mean.tolist()}))
    if len(ref_mean) != len(lr.coef_[0]):
        raise ValueError(f"{ref_path.name} does not match the model: delete it and rerun")
    cols = list(prep.get_feature_names_out())
    return {"variant": variant, "features": features, "model": model, "prep": prep,
            "coef": lr.coef_[0], "intercept": float(lr.intercept_[0]),
            "ref_mean": ref_mean, "groups": [_group_of(c) for c in cols],
            "cut_low": record["cutoff_low_to_moderate"],
            "cut_high": record["cutoff_moderate_to_high"]}


def _contribs(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, list[str], float]:
    """Per-row, per-factor log-odds contributions, factor names, baseline log-odds."""
    Z = _dense(bundle["prep"].transform(X[bundle["features"]]))
    per_col = (Z - bundle["ref_mean"]) * bundle["coef"]
    names = list(dict.fromkeys(bundle["groups"]))  # unique, keeps order
    out = np.zeros((len(X), len(names)))
    for j, g in enumerate(bundle["groups"]):
        out[:, names.index(g)] += per_col[:, j]
    baseline = bundle["intercept"] + float(bundle["ref_mean"] @ bundle["coef"])
    return out, names, baseline


def _band(p: float, b: dict) -> str:
    return "Low" if p < b["cut_low"] else "Moderate" if p < b["cut_high"] else "High"


def explain_patient(bundle: dict, patient: dict) -> dict:
    missing = [f for f in bundle["features"] if f not in patient]
    if missing:
        raise KeyError(f"Missing patient fields: {missing}")
    X = pd.DataFrame([patient])
    contrib, names, baseline = _contribs(bundle, X)
    logit = baseline + contrib[0].sum()
    p = float(1 / (1 + np.exp(-logit)))
    items = [{"feature": n, "label": LABELS.get(n, n), "value": patient.get(n),
              "log_odds": float(c), "odds_ratio_vs_average": float(np.exp(c)),
              "direction": "raises risk" if c > 0 else "lowers risk"}
             for n, c in zip(names, contrib[0])]
    items.sort(key=lambda d: abs(d["log_odds"]), reverse=True)
    return {"probability": p, "band": _band(p, bundle),
            "average_patient_risk": float(1 / (1 + np.exp(-baseline))),
            "baseline_log_odds": float(baseline), "contributions": items}


def global_importance(bundle: dict, df: pd.DataFrame) -> pd.Series:
    contrib, names, _ = _contribs(bundle, df)
    return pd.Series(np.abs(contrib).mean(axis=0), index=names).sort_values(ascending=False)


def plot_importance(imp: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    imp = imp.iloc[::-1]
    ax.barh([LABELS.get(n, n) for n in imp.index], imp.values, color="#0F5257")
    ax.set(xlabel="Average size of effect on PPH log-odds (higher = more influence)",
           title=title)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "global_importance.png", dpi=150)
    plt.close(fig)


def _print_patient(title: str, patient: dict, res: dict) -> None:
    print(f"\n{title}")
    print("  " + ", ".join(f"{k}={v}" for k, v in patient.items()))
    print(f"  Predicted risk {res['probability']:.2%} -> {res['band']} "
          f"(average patient: {res['average_patient_risk']:.2%})")
    for c in res["contributions"][:6]:
        print(f"    {c['label']:<34} {c['log_odds']:+.3f}  "
              f"(x{c['odds_ratio_vs_average']:.2f})  {c['direction']}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load("full")
    df = pd.read_csv(DATA_PATH)
    X = df[bundle["features"]]

    imp = global_importance(bundle, df)
    print("Global importance (mean |log-odds contribution|, all 56,014 patients):")
    for n, v in imp.items():
        print(f"  {LABELS.get(n, n):<34} {v:.4f}")
    plot_importance(imp, "What drives PPH risk in the model (full feature set)")

    lr = bundle["model"].named_steps["model"]
    cols = list(bundle["prep"].get_feature_names_out())
    print("\nOdds ratios (binary factors: present vs absent; numeric: per 1 SD):")
    for c, w in zip(cols, lr.coef_[0]):
        if c.startswith(("bin__", "num__")):
            print(f"  {c.split('__', 1)[1]:<24} {np.exp(w):.2f}")

    # consistency checks: explanation must reproduce the model exactly
    contrib, _, baseline = _contribs(bundle, X)
    p_expl = 1 / (1 + np.exp(-(baseline + contrib.sum(axis=1))))
    p_model = bundle["model"].predict_proba(X)[:, 1]
    print(f"\nCheck: max |explained - model probability| = {np.abs(p_expl - p_model).max():.2e}")

    # example patients (real rows of the dataset)
    p_all = pd.Series(p_model, index=df.index)
    hi = df.loc[p_all.idxmax(), bundle["features"]].to_dict()
    mid = df.loc[(p_all - p_all.median()).abs().idxmin(), bundle["features"]].to_dict()
    _print_patient("Highest-risk patient in the data:", hi, explain_patient(bundle, hi))
    _print_patient("Typical (median-risk) patient:", mid, explain_patient(bundle, mid))
    print("\nSaved: outputs/plots/global_importance.png")


if __name__ == "__main__":
    main()