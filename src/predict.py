"""
Score new customer records for churn risk using saved model artifacts.

Usage:
    python -m src.predict --input path/to/new_customers.csv --model xgboost

The input CSV must have the same raw columns as the training data (minus
the Churn label). Output is written next to the input file with a
`_scored` suffix, adding `churn_probability` and `churn_prediction` columns.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.data_preprocessing import clean_data, encode_and_scale, load_preprocessors


def score_dataframe(df: pd.DataFrame, model_name: str = "xgboost") -> pd.DataFrame:
    model = joblib.load(config.MODELS_DIR / f"{model_name}.joblib")
    encoders, scaler = load_preprocessors()

    df_raw = df.copy()

    # clean_data expects a Churn column (it encodes it); add a placeholder
    # if scoring genuinely new/unlabeled customers.
    added_placeholder = False
    if config.TARGET_COLUMN not in df_raw.columns:
        df_raw[config.TARGET_COLUMN] = "No"
        added_placeholder = True

    cleaned = clean_data(df_raw)
    encoded, _, _ = encode_and_scale(cleaned, fit=False, encoders=encoders, scaler=scaler)

    X = encoded.drop(columns=[config.TARGET_COLUMN])
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    result = df.copy()
    result["churn_probability"] = probabilities
    result["churn_prediction"] = predictions
    if added_placeholder:
        pass  # placeholder column was only used internally, not returned
    return result


def main():
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", required=True, help="Path to a CSV of new customer records.")
    parser.add_argument(
        "--model", default="xgboost",
        choices=["logistic_regression", "decision_tree", "lightgbm", "random_forest", "xgboost"],
        help="Which trained model to use for scoring.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    scored = score_dataframe(df, model_name=args.model)

    output_path = input_path.with_name(f"{input_path.stem}_scored.csv")
    scored.to_csv(output_path, index=False)
    print(f"[predict] Scored {len(scored)} records using '{args.model}'.")
    print(f"[predict] Output written to {output_path}")


if __name__ == "__main__":
    main()
