"""Evaluate PPH models with repeated stratified cross-validation.

Reads the cleaned dataset produced by preprocess.py and saves fold-level
mean and standard deviation metrics to outputs/clinical_metrics.json.
Run from the project root: python src/clinical/evaluate.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

if __package__:
    from .train import (
        DATA_PATH,
        LATE_KNOWN,
        N_REPEATS,
        N_SPLITS,
        NOT_PREDICTORS,
        SEED,
        TARGET,
        make_models,
        make_preprocessor,
    )
else:
    from train import (
        DATA_PATH,
        LATE_KNOWN,
        N_REPEATS,
        N_SPLITS,
        NOT_PREDICTORS,
        SEED,
        TARGET,
        make_models,
        make_preprocessor,
    )

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "outputs" / "clinical_metrics.json"


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error using equal-frequency bins."""
    order = np.argsort(y_prob)
    ece = 0.0
    for indices in np.array_split(order, min(n_bins, len(order))):
        if len(indices):
            ece += len(indices) / len(y_true) * abs(
                float(np.mean(y_prob[indices])) - float(np.mean(y_true[indices]))
            )
    return float(ece)


def calculate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, prevalence: float
) -> dict[str, float]:
    """Calculate discrimination, calibration, and threshold metrics for one fold."""
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e10, max_iter=1000).fit(log_odds, y_true)

    metrics = {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
        "ECE": compute_ece(y_true, y_prob),
        "Calibration_Slope": float(calibration.coef_[0, 0]),
        "Calibration_Intercept": float(calibration.intercept_[0]),
    }

    for threshold in (0.5, prevalence):
        tag = f"th_{threshold:.3f}"
        predicted = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        metrics[f"Accuracy_{tag}"] = float(accuracy_score(y_true, predicted))
        metrics[f"Sensitivity_{tag}"] = float(recall_score(y_true, predicted, zero_division=0))
        metrics[f"Specificity_{tag}"] = float(tn / (tn + fp) if tn + fp else 0.0)
        metrics[f"Precision_{tag}"] = float(precision_score(y_true, predicted, zero_division=0))
        metrics[f"F1_{tag}"] = float(f1_score(y_true, predicted, zero_division=0))

    return metrics


def evaluate_feature_set(
    df: pd.DataFrame, features: list[str], model
) -> dict[str, dict[str, float]]:
    """Return mean and standard deviation of metrics across repeated CV folds."""
    X, y = df[features], df[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    folds = []

    for train_indices, test_indices in cv.split(X, y):
        pipeline = Pipeline(
            [
                ("prep", make_preprocessor(features)),
                ("model", clone(model)),
            ]
        )
        pipeline.fit(X.iloc[train_indices], y[train_indices])
        probabilities = pipeline.predict_proba(X.iloc[test_indices])[:, 1]
        folds.append(
            calculate_metrics(y[test_indices], probabilities, float(y.mean()))
        )

    fold_metrics = pd.DataFrame(folds)
    return {
        metric: {
            "mean": float(fold_metrics[metric].mean()),
            "std": float(fold_metrics[metric].std()),
        }
        for metric in fold_metrics.columns
    }


def main(data_path: Path = DATA_PATH) -> None:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Cleaned clinical dataset not found: {data_path}. "
            "Run `python src/clinical/preprocess.py` first."
        )

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET!r}.")
    if df.empty or df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain rows from both PPH classes.")

    predictors = [column for column in df.columns if column not in NOT_PREDICTORS]
    feature_sets = {
        "full": predictors,
        "no_late": [column for column in predictors if column not in LATE_KNOWN],
    }
    for name, features in feature_sets.items():
        missing = [column for column in features if column not in df.columns]
        if missing:
            raise ValueError(f"Feature set {name!r} is missing columns: {missing}")

    print(
        f"Rows: {len(df)} | PPH: {int(df[TARGET].sum())} ({df[TARGET].mean():.2%})"
        f" | CV: {N_SPLITS}-fold x {N_REPEATS} repeats"
    )
    results = {}
    for set_name, features in feature_sets.items():
        results[set_name] = {}
        for model_name, model in make_models().items():
            print(f"Evaluating {model_name} ({set_name}, {len(features)} predictors)...")
            results[set_name][model_name] = evaluate_feature_set(df, features, model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "rows": int(len(df)),
        "pph_cases": int(df[TARGET].sum()),
        "prevalence": float(df[TARGET].mean()),
        "validation": f"Repeated stratified {N_SPLITS}-fold CV, {N_REPEATS} repeats, seed {SEED}",
        "metrics": results,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAUROC / AUPRC / Brier (mean ± SD across folds):")
    for set_name, models in results.items():
        for model_name, metrics in models.items():
            print(
                f"{set_name:<8} {model_name:<19} "
                f"{metrics['AUROC']['mean']:.4f} ± {metrics['AUROC']['std']:.4f} / "
                f"{metrics['AUPRC']['mean']:.4f} ± {metrics['AUPRC']['std']:.4f} / "
                f"{metrics['Brier_Score']['mean']:.4f} ± {metrics['Brier_Score']['std']:.4f}"
            )
    print(f"\nMetrics saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DATA_PATH,
        help="Cleaned CSV dataset (defaults to data/processed/clinical_clean.csv)",
    )
    main(parser.parse_args().data_path)
