"""
Evaluate trained models: classification metrics, a single comparison table,
and an overlaid ROC curve plot.

Usage:
    python -m src.evaluate
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)

from src import config


def load_trained_models(model_names=None):
    model_names = model_names or config.MODEL_NAMES
    return {
        name: joblib.load(config.MODELS_DIR / f"{name}.joblib")
        for name in model_names
        if (config.MODELS_DIR / f"{name}.joblib").exists()
    }


def load_test_split():
    return joblib.load(config.MODELS_DIR / "test_split.joblib")


def build_comparison_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """One row per model: accuracy, precision, recall, F1, ROC-AUC.

    This is the consolidated view the earlier per-model
    `classification_report` printouts were missing.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        })
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return table


def print_classification_reports(models: dict, X_test, y_test):
    for name, model in models.items():
        print(f"\n=== {name} ===")
        print(classification_report(y_test, model.predict(X_test)))


def plot_roc_curves(models: dict, X_test, y_test, save_path=None):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison Across Models")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[evaluate] ROC curve saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    models = load_trained_models()
    X_test, y_test = load_test_split()

    print_classification_reports(models, X_test, y_test)

    table = build_comparison_table(models, X_test, y_test)
    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(table.to_string(index=False))
    table.to_csv(config.ROOT_DIR / "reports" / "model_comparison.csv", index=False)

    plot_roc_curves(models, X_test, y_test, save_path=config.FIGURES_DIR / "roc_curves.png")
