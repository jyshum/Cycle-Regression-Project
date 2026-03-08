# Menstrual Cycle Length Prediction
### Multiple Linear Regression · Scikit-learn · Python 3.12

---

## Research Question

Can demographic and lifestyle factors — age, BMI, sleep hours, stress level,
exercise frequency, and diet — predict menstrual cycle length without relying
on personal cycle history?

---

## Result

**No — not meaningfully.**

The model achieves R² ≈ 0 and MAE of ~6.93 days, performing no better than
predicting the population mean for every individual. Exploratory data analysis
confirmed this before any modeling: every numeric feature has a near-zero
correlation with cycle length (|r| ≤ 0.04), and categorical boxplots show
heavy overlap across all exercise and diet categories. This is a valid
scientific finding — the selected features carry negligible predictive signal
for cycle length at the population level.

---

## Dataset

- **Source:** [Menstrual Cycle Data with Factors — Kaggle](https://www.kaggle.com/datasets/akshayas02/menstrual-cycle-data-with-factors-dataset)
- **Size:** 895 rows · 100 unique users · 6–12 cycles per user (mean: ~9)
- **Structure:** Longitudinal — multiple cycle records per person
- **Nulls:** None — no imputation required

---

## Features

| Feature | Type | Preprocessing |
|---|---|---|
| Age | Numeric | Z-score standardized |
| BMI | Numeric | Z-score standardized |
| Stress Level (1–5) | Numeric | Z-score standardized |
| Sleep Hours | Numeric | Z-score standardized |
| Period Length | Numeric | Z-score standardized |
| Exercise Frequency | Categorical | One-hot (baseline: Low) |
| Diet | Categorical | One-hot (baseline: Vegetarian) |

**Target:** `Cycle Length` (days, range: 25–50, mean: 37.4, std: 7.5)

---

## Exploratory Data Analysis

EDA was conducted in `notebooks/01_eda.ipynb` before any modeling.

**Target distribution** — Cycle length is spread uniformly across 25–50 days
with no outliers. The lack of a dominant central peak means the mean is a
surprisingly competitive baseline — there is no obvious cluster for a model
to exploit.

**Numeric features** — All five numeric features show near-zero Pearson
correlation with cycle length (highest: Sleep Hours at r = 0.04). Scatterplots
confirm flat trend lines across every feature with no hidden non-linear signal.
Stress Level and Period Length show discrete vertical striping, confirming
their ordinal nature.

**Categorical features** — Exercise Frequency and Diet are reasonably balanced
across categories. Boxplots show heavy overlap in cycle length distributions
across all exercise and diet groups — no category meaningfully separates
cycle lengths from the rest.

**Correlation heatmap** — The bottom row (Cycle Length correlations) is
entirely near-zero, confirming that no available feature has a linear
relationship with the target worth modeling.

---

## Methodology

### Data Splitting — GroupShuffleSplit
The dataset contains multiple rows per user. A naive random split would place
the same user in both train and test — leaking information and inflating
apparent performance. `GroupShuffleSplit` ensures all rows for a given user
go entirely to one split. Row counts are approximate rather than exact because splitting by user identity(6–12 rows each) cannot guarantee a precise row-level ratio.
```
Train: 706 rows (80 users)
Test:  189 rows (20 users)
```

### Baseline Model
Before training, a naive baseline is established: predict the mean cycle
length of the training set for every test sample. Any model that cannot beat
this has learned nothing useful.
```
Baseline mean prediction: 37.27 days
```

### Standardization
Numeric features are standardized using `StandardScaler` fit **on training
data only**. The same fitted scaler transforms the test set. Fitting on the
full dataset would leak test distribution information into training.

Categorical features are one-hot encoded with one category dropped per feature
to avoid the dummy variable trap — perfect multicollinearity where one column
is always predictable from the others, destabilizing model coefficients.
Baselines were chosen explicitly (`drop_first=False`) rather than
alphabetically, preserving interpretability: coefficients represent effects
*compared to Low exercise and Vegetarian diet*.

---

## Results

| Metric | Baseline | Linear Regression | Improvement |
|---|---|---|---|
| MAE (days) | 6.8893 | 6.9299 | -0.0406 ❌ |
| RMSE (days) | 7.8009 | 7.8385 | -0.0376 ❌ |
| R² | ~0.00 | -0.013 | -0.010 ❌ |

The linear regression model does not outperform the mean baseline on any
metric. Predictions cluster tightly between 35.5–38.5 days for all test
samples — the model fails to differentiate meaningfully between individuals.
This outcome was consistent with EDA findings, where all feature correlations
with cycle length were effectively zero before modeling began.

---

## Residual Plot

![Residual Plot](reports/residual_plot.png)

Residuals are large (±10–12 days) and symmetric around zero. The narrow
prediction range (3 days) relative to the actual target range (25 days)
confirms the model found no meaningful signal. The symmetry rules out
systematic bias — the model is not consistently over or under predicting,
it simply cannot differentiate between individuals. The x-axis spans only 3 days (35.5–38.5) despite the actual target ranging
25 days (25–50) — the model assigns nearly identical predictions to all
individuals, further confirming the absence of meaningful feature signal.

---

## Coefficient Table

| Feature | Coefficient | Direction |
|---|---|---|
| Diet_High Sugar | +0.818 | longer cycle |
| Diet_Balanced | +0.727 | longer cycle |
| Exercise Frequency_High | +0.693 | longer cycle |
| Exercise Frequency_Moderate | +0.471 | longer cycle |
| Diet_Low Carb | +0.335 | longer cycle |
| Age | -0.330 | shorter cycle |
| BMI | +0.297 | longer cycle |
| Sleep Hours | +0.285 | longer cycle |
| Period Length | -0.200 | shorter cycle |
| Stress Level | +0.148 | longer cycle |

Coefficients are on a standardized scale. No single feature exerts meaningful
influence — the largest coefficient (0.818) is small relative to the target
standard deviation of 7.5 days. Diet and exercise dominate weakly over
demographic features, consistent with their slightly larger boxplot separation
observed in EDA.

---

## Project Structure
```
cycle-regression-project/
├── data/
│   ├── raw/                  # original Kaggle CSV (unmodified)
│   └── processed/            # cleaned outputs
├── notebooks/
│   └── 01_eda.ipynb          # EDA — distributions, correlations, boxplots
├── src/
│   ├── preprocess.py         # cleaning, encoding, splitting, scaling
│   ├── train.py              # baseline + linear regression training
│   └── evaluate.py           # metrics, comparison, error analysis
├── tests/
│   └── test_preprocess.py    # 11 tests covering shape, leakage, encoding
├── reports/
│   ├── metrics.json          # saved evaluation results
│   ├── residual_plot.png     # visual error analysis
│   └── coefficients.csv      # feature influence table
├── models/
│   └── linear_regression.pkl # saved trained model
├── config.py                 # all paths, column names, constants
├── main.py                   # single entry point
└── requirements.txt
```

---

## How To Run
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place Kaggle CSV in data/raw/menstrual_cycle_data.csv

# 4. Run full pipeline
python main.py

# 5. Run tests
python -m pytest tests/test_preprocess.py -v

# 6. Open EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

---

## Limitations

- **No cycle history features** — prior cycle length is the strongest known
  predictor of next cycle length. This project intentionally excludes it to
  isolate the contribution of lifestyle and demographic factors alone.
- **Small unique user count** — 100 users limits generalizability.
- **Self-reported data** — stress, sleep, and diet labels may lack precision.
- **Linear model only** — non-linear relationships (e.g. Age², BMI²) not explored.
- **Synthetic dataset** — the Kaggle dataset may not reflect real-world
  population distributions, which could affect the generalizability of findings.

---

## Future Work

- Test polynomial features (Age², BMI²) to capture non-linear effects
- Compare R² against a model that includes previous cycle length — quantify
  how much history improves prediction
- Try Ridge and Lasso regression to assess whether regularization helps
- Validate findings on a real-world clinical dataset

---

## Dependencies

- Python 3.12
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pytest
- jupyter