# Responsible for:
#   1. computing baseline metrics
#   2. computing linear regression metrics
#   3. comparing both models (baseline & actual)
#   4. savinf results to reports/metrics.json

import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

def compute_metrics(y_true, y_pred, label: str) -> dict:
    """
    Computes MAE, RMSE, and R² for a set of predictions

    MAE  - Mean Absolute Error
            Average error in days. Most interpretable metric.
            "On average, my model is off by X days."

    RMSE - Root Mean Squared Error
            Like MAE but squares errors first, then roots.
            Penalizes large errors more heavily than small ones.
            If RMSE >> MAE, you have outlier predictions worth investigating.

    R²   - Coefficent of Determination
            % of variance in cycle length explained by the model.
            R² = 1.0 -> perfect. R² = 0.0 -> no better than mean baseline
            R² < 0.0 -> worse than predicting the mean
    
    Returns a dict so results can be compared and saved cleanly
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    return {
        "label": label,
        "MAE":   round(float(mae), 4),
        "RMSE":  round(float(rmse), 4),
        "R2":    round(float(r2), 4)
    }

def compare_models(baseline_metrics: dict, model_metrics: dict) -> dict:
    """
    Computes how much the linear regression improved over baseline

    MAE improvement:
        Positive = model is better (lower error)
        Negative = model is worse than just predicting the mean

    R² improvement:
        How much more variance the model explains vs baseline.
        Baseline R² is always exactly 0.0 by definition

    This comparison is the honest answer to whether the model learnt anything
    """

    mae_improvement  = baseline_metrics["MAE"] - model_metrics["MAE"]
    rmse_improvement = baseline_metrics["RMSE"] - model_metrics["RMSE"]
    r2_improvement   = model_metrics["R2"] - baseline_metrics["R2"]

    return {
        "MAE_improvement":  round(mae_improvement, 4),
        "RMSE_improvement": round(rmse_improvement, 4),
        "R2_improvement":   round(r2_improvement, 4),
    }

def print_results(baseline_metrics: dict, model_metrics: dict, improvement: dict) -> None:
    """
    Prints a clean, readable comparison to the terminal
    Designed to be scannable at a glance
    """
    print("\n" + "="*55)
    print(" MODEL EVALUATION RESULTS")
    print("="*55)

    print(f"\n{'Metric':<10} {'Baseline':>12} {'Linear Reg':>12} {'Improvement':>12}")
    print("-"*55)

    # MAE - pos improv is good
    mae_flag = "✅" if improvement["MAE_improvement"] > 0 else "❌"
    print(
        f"{'MAE':<10}"
        f"{baseline_metrics['MAE']:>12.4f}"
        f"{model_metrics['MAE']:>12.4f}"
        f"{improvement['MAE_improvement']:>+11.4f} {mae_flag}"
    )

    # RMSE - lower is better
    rmse_flag = "✅" if improvement['RMSE_improvement'] > 0 else "❌"
    print(
        f"{'RMSE':<10}"
        f"{baseline_metrics['RMSE']:>12.4f}"
        f"{model_metrics['RMSE']:>12.4f}"
        f"{improvement['RMSE_improvement']:>+11.4f} {rmse_flag}"
    )

    # R² — higher is better
    r2_flag = "✅" if improvement["R2_improvement"] > 0 else "❌"
    print(
    f"{'R²':<10}"
    f"{baseline_metrics['R2']:>12.4f}"
    f"{model_metrics['R2']:>12.4f}"
    f"{improvement['R2_improvement']:>+11.4f} {r2_flag}"
    )

    print("="*55)

    # writeup
    print("\n📋 VERDICT")
    if improvement["MAE_improvement"] > 0:
        print(
            f"  Model beats baseline by {improvement['MAE_improvement']:.4f} days MAE.\n"
            f"  R² = {model_metrics['R2']:.4f} - model explains"
            f"{model_metrics['R2']*100:.1f}% of variance in cycle length."
        )
        if model_metrics["R2"] < 0.10:
            print(
                "   ⚠️  R² is low. Features carry limited predictive signal.\n"
                "   This is a valid scientific finding, not a code error."
            )
    else:
        print(
            "   ❌ Model does NOT beat baseline.\n"
            "   Predicting the mean outperforms the regression model.\n"
            "   This suggests features have very weak signal for cycle length."
        )
    print()

def save_results(
        baseline_metrics: dict,
        model_metrics: dict,
        improvement: dict
) -> None:
    """
    Saves all results to reports/metrics.json
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "baseline": baseline_metrics,
        "linear_regression": model_metrics,
        "improvement": improvement,
    }

    with open(config.METRICS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {config.METRICS_FILE}")

def evaluate(baseline, model, X_test, y_test) -> dict:
    """
    evaluation function - called by main.py
    steps:
        1. generate predictions from both models on the test set only
        2. compute metrics for both
        3. compare them
        4. print results
        5. save to disk
    """
    
    # baseline predicts mean
    baseline_preds = baseline.predict(np.zeros((len(y_test), 1)))
    model_preds    = model.predict(X_test)

    baseline_metrics = compute_metrics(y_test, baseline_preds, "Baseline")
    model_metrics    = compute_metrics(y_test, model_preds, "Linear Regression")
    improvement      = compare_models(baseline_metrics, model_metrics)

    print_results(baseline_metrics, model_metrics, improvement)
    save_results(baseline_metrics, model_metrics, improvement)

    return {
        "baseline": baseline_metrics,
        "model":    model_metrics,
        "improvement": improvement,
    }

def analyze_errors(model, X_test, y_test, feature_names: list) -> None:
    """
    Performs two analyses that explain why the model performs as it does:

    1. Residual plot
        r = y - yhat for each test sample
        good model has residuals scattered randomly around 0
        patterns in residuals (e.g. fan shape, curve) reveal model failures:
            - Fan shape -> variance increases with predictio
            - Curve     -> a linear model is missing non-linear relationships
            - Bias      -> residuals above or below 0

    2. Coefficient Table
        shows each feature's coefficient - how much a 1std dev change in that feature
        shifts the predicted cycle length. Larger abs value = stronger influence
        on prediction
        - sign tells direction: positive = longer cycle, negative = shorter
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — saves to file, no popup
    import matplotlib.pyplot as plt
    import pandas as pd

    # ── Residual Plot ─────────────────────────────────────────────────────────

    model_preds = model.predict(X_test)

    # residual = what actually happened minus what model predicted
    residuals = np.array(y_test) - model_preds

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(model_preds, residuals, alpha=0.5, edgecolors="k", linewidths=0.3)

    # horizontal line at 0 — perfect predictions would all sit here
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero error")

    ax.set_xlabel("Predicted Cycle Length (days)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot — Linear Regression vs Actual Cycle Length")
    ax.legend()

    plt.tight_layout()

    residual_path = config.REPORTS_DIR / "residual_plot.png"
    plt.savefig(residual_path, dpi=150)
    plt.close()
    print(f"Residual plot saved to: {residual_path}")

    # ── Coefficient Table ─────────────────────────────────────────────────────

    # zip pairs each feature name with its learned coefficient
    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_.round(4),
    })

    # sort by absolute value — strongest predictors at the top
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).drop(columns="Abs")
    coef_df["Direction"] = coef_df["Coefficient"].apply(
        lambda x: "longer cycle" if x > 0 else "shorter cycle"
    )

    print("\n📊 COEFFICIENT TABLE (sorted by influence)")
    print("-" * 50)
    print(coef_df.to_string(index=False))
    print("-" * 50)

    # save coefficient table as CSV for portfolio
    coef_path = config.REPORTS_DIR / "coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Coefficient table saved to: {coef_path}")


if __name__ == "__main__":
    from src.preprocess import preprocess
    from src.train import train

    X_train, X_test, y_train, y_test, feature_names = preprocess()
    baseline, model = train(X_train, y_train)
    evaluate(baseline, model, X_test, y_test)
    analyze_errors(model, X_test, y_test, feature_names)

