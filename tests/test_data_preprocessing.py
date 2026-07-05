"""
Tests for src/data_preprocessing.py.

Run with: pytest tests/
These only depend on pandas/scikit-learn (already required for the whole
project), so they run in any environment that can import src.data_preprocessing.
"""
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    clean_data,
    encode_and_scale,
    get_train_test_split,
    load_raw_data,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


def test_raw_data_loads(raw_df):
    assert len(raw_df) > 0
    assert config.TARGET_COLUMN in raw_df.columns


def test_clean_data_drops_id_and_encodes_target(raw_df):
    cleaned = clean_data(raw_df)
    assert config.ID_COLUMN not in cleaned.columns
    assert set(cleaned[config.TARGET_COLUMN].unique()) <= {0, 1}


def test_clean_data_total_charges_is_numeric_with_no_nulls(raw_df):
    cleaned = clean_data(raw_df)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_encode_and_scale_produces_all_numeric_features(raw_df):
    cleaned = clean_data(raw_df)
    encoded, encoders, scaler = encode_and_scale(cleaned, fit=True)
    feature_cols = [c for c in encoded.columns if c != config.TARGET_COLUMN]
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in feature_cols)
    assert set(encoders.keys()) == set(config.CATEGORICAL_COLUMNS)


def test_train_test_split_is_stratified(raw_df):
    cleaned = clean_data(raw_df)
    encoded, _, _ = encode_and_scale(cleaned, fit=True)
    X_train, X_test, y_train, y_test = get_train_test_split(encoded)

    # Stratified split should keep churn rate within a small tolerance
    # between train and test.
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_shapes_are_consistent(raw_df):
    cleaned = clean_data(raw_df)
    encoded, _, _ = encode_and_scale(cleaned, fit=True)
    X_train, X_test, y_train, y_test = get_train_test_split(encoded)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
