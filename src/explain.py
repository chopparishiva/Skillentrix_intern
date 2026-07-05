"""
Model explainability via SHAP.

Usage:
    python -m src.explain

Notes
-----
`shap.TreeExplainer(...).shap_values(X)` return shape is inconsistent across
shap/xgboost/lightgbm versions for binary classification: some versions
return a single 2D array (n_samples, n_features), others return a list of
two arrays (one per class). `_extract_positive_class_shap_values` below
normalizes both cases so this script doesn't break silently depending on
which library versions are installed.
"""
import joblib
import numpy as np
import shap

from src import config


def _extract_positive_class_shap_values(shap_values):
    """Normalize shap_values to a single 2D array for the positive class."""
    if isinstance(shap_values, list):
        # Older API: list of arrays, one per class. Index 1 = "churn" class.
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Some versions return (n_samples, n_features, n_classes).
        return shap_values[:, :, 1]
    return shap_values


def explain_model(model_name: str = "xgboost", max_display: int = 15):
    model = joblib.load(config.MODELS_DIR / f"{model_name}.joblib")
    X_test, _ = joblib.load(config.MODELS_DIR / "test_split.joblib")

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test)
    shap_values = _extract_positive_class_shap_values(raw_shap_values)

    shap.summary_plot(
        shap_values, X_test, plot_type="bar", max_display=max_display, show=False
    )
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[explain] SHAP plots for '{model_name}' saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    explain_model("xgboost")
