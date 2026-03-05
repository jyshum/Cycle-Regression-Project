# tests for src/preprocess.py
# Run with:
#   python -m pytest tests/test_preprocess.py -v
# Each test verifies one step of the pipeline
# if any of these fail, something is broken

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.preprocess import preprocess
import config

@pytest.fixture(scope="module")
def preprocessed_data():
    """
    Runs the full pipeline once and shares the results across all tests.
    scope="module" means preprocess() is only called once per test run,
        - keeps test fast
    """
    X_train, X_test, y_train, y_test, feature_names = preprocess()
    return X_train, X_test, y_train, y_test, feature_names

# tests

def test_train_test_split_sizes(preprocessed_data):
    """
    Total rows in train + test must equal the full dataset (895 rows)
    If this fails, rows are being dropped somewhere
    """
    X_train, X_test, y_train, y_test, _ = preprocessed_data
    assert len(X_train) + len(X_test) == 895, (
        f"Expected 895 total rows, got {len(X_train) + len(X_test)}"
    )

def test_train_test_ratio(preprocessed_data):
    """
    Test set should be approximately 20% of total data.
    We allow ±3% tolerance because GroupShuffleSplit splits by user
        - not individual rows
        - 80/20 is not guaranteed
    """
    X_train, X_test, _, _, _ = preprocessed_data
    total = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.17 <= test_ratio <= 0.23, (
        f"Expected -20% test split, got {test_ratio:.2%}"
    )

def test_feature_count(preprocessed_data):
    """
    expect exactly 10 feature columns after preproccesing
    5 numeric + 2 excercise dummies + 3 diet dummies = 10
    if this fales, encoding or dropping logic changed
    """
    X_train, _, _, _, feature_names = preprocessed_data
    assert X_train.shape[1] == 10, (
        f"Expected 10 features, got {X_train.shape[1]}: {feature_names}"
    )

# leakage tests

def test_target_not_in_features(preprocessed_data):
    """
    the target column must never appear in X_train or X_test.
    """
    X_train, X_test, _, _, _ = preprocessed_data
    assert config.TARGET_COL not in X_train.columns, "Target column found in X_train"
    assert config.TARGET_COL not in X_test.columns, "Target column found in X_test"

def test_user_id_not_in_features(preprocessed_data):
    """
    Usrr ID must be dropped -> no predictive value; useless
    """
    X_train, X_test, _, _, _ = preprocessed_data
    assert config.USER_ID_COL not in X_train.columns, "User ID found in X_train"
    assert config.USER_ID_COL not in X_test.columns, "User ID found in X_test"

# null tests

def test_no_nulls_in_X_train(preprocessed_data):
    """
    No NaN values should exist in training features.
    A model trained on NaNs will produce unreliable coefficients
    """
    X_train, _, _, _, _ = preprocessed_data
    null_counts = X_train.isnull().sum()
    assert null_counts.sum() == 0, (
        f"NaNs found in X_train:\n{nulls_count[null_counts > 0]}"
    )

def test_no_nulls_in_y_train(preprocessed_data):
    """
    same thing as def above, just in target vector
    """
    _, _, y_train, _, _ = preprocessed_data
    assert y_train.isnull().sum() == 0, "NaNs found in y_train"

# encoding tests

def test_expected_feaure_names(preprocessed_data):
    """
    - verifies the exact set of expected feature coums exists
    - catches baseline encoding errors
    """
    _, _, _, _, feature_names = preprocessed_data
    expected = {
        "Age", "BMI", "Stress Level", "Sleep Hours", "Period Length",
        "Exercise Frequency_High", "Exercise Frequency_Moderate",
        "Diet_Balanced", "Diet_High Sugar", "Diet_Low Carb"
    }
    actual = set(feature_names)
    assert actual == expected, (
        f"Feature mismatch. \nExpected: {sorted(expected)}\nGot:{sorted(actual)}"
    )

def test_baseline_categories_dropped(preprocessed_data):
    """
    checks that our chosen baselines were dropped
    Excercise baseline = Low, Diet baseline = Vegetarian
    """
    _, _, _, _, feature_names = preprocessed_data
    assert "Excercise Frequency_Low" not in feature_names, \
        "Baseline 'Excercise Frequency_Low' should be dropped"
    assert "Diet_Vegetarian" not in feature_names, \
        "Baseline 'Diet_Vegetarian' should be dropped"

def test_one_hot_columns_are_binary(preprocessed_data):
    """
    one-hot encoded columns must only contain 0.0 or 1.0
    """ 
    X_train, _, _, _, _ = preprocessed_data
    ohe_cols = [c for c in X_train.columns if 
                c.startswith("Excercise Frequency_") or c.startswith("Diet_")]
    for col in ohe_cols:
        unique_vals = set(X_train[col].unique())
        assert unique_vals <= {0.0, 1.0}, (
            f"Column '{col}' contains non-binary values: {unique_vals}"
        )

# Scaling tests

def test_numeric_cols_are_standardized(preprocessed_data):
    """
    After StandardScaler, numeric columns should have mean = 0
    and standard deviation = 1
    - allow tolerance of 0.05 for floating point variation
    """
    X_train, _, _, _, _ = preprocessed_data
    for col in config.NUMERIC_COLS:
        mean = X_train[col].mean()
        std = X_train[col].std()
        assert abs(mean) < 0.05, f"Column '{col}' mean is {mean:.4f}, expected ~0"
        assert abs(std - 1.0) < 0.05, f"Column '{col}' std is {std:.4f}, expected ~1"
