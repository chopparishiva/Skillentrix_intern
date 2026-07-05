"""
Train all churn-prediction models and persist them to disk.

Usage:
    python -m src.train

All models are seeded with `config.RANDOM_STATE` so results are
reproducible run to run. Trained models and preprocessors are written to
`models/` as .joblib files for reuse by evaluate.py, explain.py, and
predict.py without retraining.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import build_dataset, save_preprocessors

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE
    ),
    "decision_tree": lambda: DecisionTreeClassifier(
        criterion="entropy", max_depth=4, random_state=config.RANDOM_STATE
    ),
    "lightgbm": lambda: LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=config.RANDOM_STATE
    ),
    "xgboost": lambda: XGBClassifier(
        eval_metric="logloss", random_state=config.RANDOM_STATE
    ),
}
assert set(MODEL_REGISTRY) == set(config.MODEL_NAMES), (
    "MODEL_REGISTRY keys must match config.MODEL_NAMES exactly."
)


def train_all_models():
    X_train, X_test, y_train, y_test, encoders, scaler = build_dataset()

    trained = {}
    for name, factory in MODEL_REGISTRY.items():
        model = factory()
        model.fit(X_train, y_train)
        trained[name] = model
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        print(f"[train] {name} trained and saved.")

    save_preprocessors(encoders, scaler)
    print(f"[train] preprocessors saved to {config.MODELS_DIR}")

    # Persist the split so evaluate.py/explain.py score on the exact same
    # held-out data the models were trained against.
    joblib.dump((X_test, y_test), config.MODELS_DIR / "test_split.joblib")

    return trained, X_test, y_test


if __name__ == "__main__":
    train_all_models()
