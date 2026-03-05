# Notes:
# central configuration for the cycle regression project
# paths, column names, hyperparameters
# nothing in src/ shuld hardcode strings or numeric constants

from pathlib import Path

# Paths 

ROOT_DIR        = Path(__file__).resolve().parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models"
REPORTS_DIR     = ROOT_DIR / "reports"

RAW_DATA_FILE   = DATA_RAW_DIR / "menstrual_cycle_data.csv"
MODEL_FILE      = MODELS_DIR / "Linear_regression.pkl"
METRICS_FILE    = REPORTS_DIR / "metrics.json"

# Column Names

USER_ID_COL     = "User ID"
TARGET_COL      = "Cycle Length"

NUMERIC_COLS   = [
    "Age",
    "BMI",
    "Stress Level",
    "Sleep Hours",
    "Period Length",
]

CATEGORICAL_COLS = [
    "Exercise Frequency",
    "Diet",
]

DROP_COLS       = [
    "User ID",
    "Cycle Start Date",
    "Next Cycle Start Date",
    "Symptoms",
]

# Split Settings

TEST_SIZE       = 0.20
RANDOM_SEED     = 42

# ONE-HOT Encoding Refernece
# Exercise Freqiency: baseline = low
#       Exercise Frequency_Moderate, Exercise Frequency_High
# Diet: baseline = Vegetarian
#       Diet_Balanced, Diet_High Sugar, Diet_Low Carb