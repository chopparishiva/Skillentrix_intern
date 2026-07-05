"""
Central configuration: paths and constants used across the pipeline.
Keeping these in one place avoids magic numbers/strings scattered across
scripts and makes the pipeline reproducible.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_DATA_PATH = DATA_DIR / "telco_customer_churn.csv"

for _dir in (MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 44
TEST_SIZE = 0.25

# --- Feature groups --------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Canonical list of model names used across train/evaluate/explain/predict.
# Kept here (not in train.py) so modules that only need the *names* don't
# have to import train.py and pull in lightgbm/xgboost as a side effect.
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "lightgbm",
    "random_forest",
    "xgboost",
]
