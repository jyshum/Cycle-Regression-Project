# Single entry point for full pipeline
# run entire project with: python main.py
#   1. preprocess -> clean, encode, split, scale
#   2. Train -> fit baseline + linear regression
#   3. Evaluate -> compute metrics, compare, save results
#   4. analyze -> residual plot + coeff table

from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate, analyze_errors

def main():
    print("\n Step 1 - Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    print("\n Step 2 - Training...")
    baseline, model = train(X_train, y_train)

    print("\n Step 3- Evaluating...")
    evaluate(baseline, model, X_test, y_test)

    print("\n Step 4 - Error Analysis...")
    analyze_errors(model, X_test, y_test, feature_names)

    print("\n✅ Pipeline complete. Check reports/ for saved outputs. \n")

if __name__ == "__main__":
    main()