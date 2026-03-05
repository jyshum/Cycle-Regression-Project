# Responsible for:
#   1. Loading raw data
#   2. Dropping irrelevant columns
#   3. One-Hot encoding categorical features
#   4. Splitting by User ID (GroupShuffleSplit -> no leakage)
#   5. Standardizing numeric features (fit on train only)
# Returns: X_train, X_test, y_train, y_test, feature_names

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


def load_raw_data() -> pd.DataFrame:
    # Loads the raw CSV from the path defined in config
    # Does nothing else; loading and cleaning are separate responsibilities

    df = pd.read_csv(config.RAW_DATA_FILE)
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drops columns we have decided not to use
    # see config.DROP_COLS

    return df.drop(columns=config.DROP_COLS)

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # one-hot encodes categorical columns stated in config.CATEGORICAL_COLS
    """
    Rules:
    - Baselines: Excercise Frequency=low, Diet=Vegetarian
    - One-Hot columns are NOT standardized ( 0/1 )
    """
    df = pd.get_dummies(
        df,
        columns=config.CATEGORICAL_COLS,
        drop_first=False,
        dtype=float     # keep columns as float for sklearn 
    )

    df = df.drop(columns=[
        "Exercise Frequency_Low", # baseline: Low
        "Diet_Vegetarian",        # baseline: Vegetarian
    ])
    return df

def split_by_user(
        df: pd.DataFrame, 
        groups: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into traina dn test sets so that ALL rows
    belonging to a given User ID go entirely to one split; never both
            - prevents data leakage that would occur with train_test_split
            when the same user has multiple rows (duplicates)
    returns (df_train, df_test)
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    train_idx, test_idx = next(splitter.split(df, groups=groups))

    return df.iloc[train_idx], df.iloc[test_idx]

def standardize_numeric(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits a StandardScaler on training data only, then transforms
    both train and test sets

    rule: never fit the scaler on test data or the full dataset
        - leaks test distribution info into training
        - inflates model performance
    
    Returns: (df_train_scaled, df_test_scaled, fitted_scaler)
    the scaler is returned so it can be reused at inference time if needed
    """
    scaler = StandardScaler()

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[config.NUMERIC_COLS] = scaler.fit_transform(df_train[config.NUMERIC_COLS])
    df_test[config.NUMERIC_COLS] = scaler.transform(df_test[config.NUMERIC_COLS])

    return df_train, df_test, scaler

def preprocess() -> tuple:
    """
    pipeline function; calls each step in order
    this is what main.py and tests will call

    Returns:
        X_train, X_test   - feature matrices (pd.DataFrame)
        y_train, y_test   - target vectors (pd.Series)
        feature_names     - list of column names in X 
    """

    # Step 1 - load
    df = load_raw_data()
    
    # Step 2 - save User ID before dropping (for GroupShuffleSplit)
    groups = df[config.USER_ID_COL]

    # Step 3 - drop unused columns
    df = drop_unused_columns(df)

    # Step 4 - encode categoricals
    df = encode_categoricals(df)

    # Step 5- separate target from features
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    # Step 6 - User-aware split
    X_train, X_test = split_by_user(X, groups)
    y_train = y.iloc[X_train.index] if hasattr(X_train.index, '__iter__') else y.loc[X_train.index]
    y_test = y.iloc[X_test.index] if hasattr(X_test.index, '__iter__') else y.loc[X_test.index]

    # Step 7 - standardize numeric columns (fit on train only)
    X_train, X_test, _ = standardize_numeric(X_train, X_test)
    feature_names = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names
