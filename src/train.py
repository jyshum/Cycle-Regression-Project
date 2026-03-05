# Responsibile for:
#   1. training a naive mean baseline (the bar)
#   2. training a multiple linear regression model
#   3. Saving the trained model to disk

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

def train_baseline(y_train: np.ndarray) -> DummyRegressor:
    """
    Trains a naive baseline that predicts the mean of the y_train
    for every single input
        - minimum model that has to be beat
    """
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)

    return baseline

def train_linear_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
) -> LinearRegression:
    """
    Trains the multiple linea regression model:
        cycle_length = β₀ + β₁(Age) + β₂(BMI) + ... + βₚ(feature_p)
    LinearRegression() finds the coefficients (β values) that minimize
    the sum of squared errors between predictions and actual cycle lengths
    on the training set. 
    - Ordinary Least Squares (0LS)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def save_model(model: LinearRegression) -> None:
    """
    serializes the trained model to disk using pickle
    - save path comes from config.py
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {config.MODEL_FILE}")

def load_model() -> LinearRegression:
    """
    Loads a previosuly saved model from disk.
    Used by evaluate.py so it dosen't need to retrain.
    """
    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    return model

def train(X_train, y_train) -> tuple[DummyRegressor, LinearRegression]:
    """
    training function; called by main.py and evaluate.py
        1. train baselin
        2. train linear regression
        3. save LR model to disk
    returns both so evaluate.py can compare them
    """
    print("Training baseline model...")
    baseline = train_baseline(y_train)

    print("Training linear regression model...")
    model = train_linear_regression(X_train, y_train)

    print("Saving linear regression model...")
    save_model(model)

    return baseline, model

if __name__ == "__main__":
    # Quick smoke test — run this file directly to confirm training works
    # python src/train.py
    from src.preprocess import preprocess

    X_train, X_test, y_train, y_test, feature_names = preprocess()
    baseline, model = train(X_train, y_train)

    print(f"\nBaseline mean prediction: {baseline.predict(np.zeros((1,1)))[0]:.2f} days")
    print(f"Model coefficients: {dict(zip(feature_names, model.coef_.round(3)))}")
    print(f"Model intercept: {model.intercept_:.4f}")


