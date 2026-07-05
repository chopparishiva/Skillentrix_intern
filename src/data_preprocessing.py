"""
Data loading and preprocessing for the Telco Customer Churn dataset.

This module turns the ad-hoc cleaning steps from the original exploratory
notebook into reusable, testable functions so the same logic can run in
training, evaluation, and inference without copy-pasting notebook cells.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the core cleaning steps identified during EDA.

    - Drop the customer ID (not predictive).
    - Coerce TotalCharges to numeric (11 rows contain blank strings in the
      raw data) and impute the resulting NaNs with the column median.
    - Encode the target as a binary integer.
    - Treat SeniorCitizen as a categorical flag rather than a numeric column.
    """
    df = df.copy()

    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

    return df


def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
):
    """Label-encode categorical columns and standard-scale numerical columns.

    Parameters
    ----------
    fit : if True, fit new encoders/scaler on `df` (training time).
          if False, reuse the provided `encoders`/`scaler` (inference time)
          so new data is transformed consistently with training data.
    """
    df = df.copy()
    encoders = encoders or {}

    for col in config.CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    if fit:
        scaler = StandardScaler()
        df[config.NUMERICAL_COLUMNS] = scaler.fit_transform(df[config.NUMERICAL_COLUMNS])
    else:
        df[config.NUMERICAL_COLUMNS] = scaler.transform(df[config.NUMERICAL_COLUMNS])

    return df, encoders, scaler


def get_train_test_split(df: pd.DataFrame):
    """Split features/target using the project-wide random state."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )


def build_dataset():
    """End-to-end: raw CSV -> cleaned, encoded, scaled train/test split.

    Returns X_train, X_test, y_train, y_test, encoders, scaler
    """
    df = load_raw_data()
    df = clean_data(df)
    df, encoders, scaler = encode_and_scale(df, fit=True)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    return X_train, X_test, y_train, y_test, encoders, scaler


def save_preprocessors(encoders: dict, scaler: StandardScaler):
    joblib.dump(encoders, config.MODELS_DIR / "label_encoders.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")


def load_preprocessors():
    encoders = joblib.load(config.MODELS_DIR / "label_encoders.joblib")
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    return encoders, scaler
