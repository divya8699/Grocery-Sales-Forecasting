# 🛒 Grocery Sales Forecasting — Linear Regression & KNN

A time-series sales forecasting notebook that predicts daily grocery sales for an upcoming 3-month period using Linear Regression and K-Nearest Neighbors Regressor, each evaluated with and without RobustScaler. The pipeline covers normality testing, outlier detection, EDA with interactive Plotly charts, and a consolidated 4-model prediction comparison on the holdout test set.

---

## 📁 Dataset Requirements

This notebook requires **two CSV files** in the same directory as the notebook:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `Train.csv` | 692 | 2 | Daily grocery sales over ~2 years |
| `Test.csv` | ~90 | 2 | Future days for which sales must be forecast (days 693+) |

> **Source:** [MachineHack — Grocery Store Sales Forecasting Challenge](https://machinehack.com/hackathons/grocery_store_forecasting_challenge)  
> Download both files and place them in the same directory as the notebook.

### Dataset Schema

| Column | Type | Description |
|---|---|---|
| `Day` | int | Sequential day number (Train: 1–692; Test: 693+) |
| `GrocerySales` | float | Daily sales in currency units |

### Training Data Statistics

| Metric | Value |
|---|---|
| Total rows | 692 |
| Null values | None |
| Mean daily sales | 8,564.73 |
| Std deviation | 428.82 |
| Min | 6,766.37 |
| Max | 9,290.02 |
| Normality (D'Agostino test) | **Not normal** (p = 3.7×10⁻³³) |

---

## 🛠️ Dependencies

Install all required packages via pip:

```bash
pip install numpy pandas matplotlib seaborn scipy plotly scikit-learn
```

| Library | Version (recommended) | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Array reshaping, numerical ops |
| `pandas` | ≥ 1.3 | Data loading, melting, concat |
| `matplotlib` | ≥ 3.4 | Inline plots |
| `seaborn` | ≥ 0.11 | Boxplot, KDE, residual histograms |
| `scipy` | ≥ 1.7 | `normaltest()`, `kurtosistest()` |
| `plotly` | ≥ 5.0 | Interactive treemap, scatter with OLS trendline |
| `scikit-learn` | ≥ 0.24 | `train_test_split`, `RobustScaler`, `LinearRegression`, `KNeighborsRegressor`, MSE/MAE metrics |

**Python version:** 3.7+

> ⚠️ `plotly` renders interactive HTML charts in Jupyter. In VS Code or non-browser environments, install the `nbformat` and `nbconvert` extensions or use `fig.write_html()` to save charts as HTML files.

---

## ▶️ How to Execute

### Option 1 — Jupyter Notebook (recommended)

```bash
pip install notebook
jupyter notebook Grocery_sales_analysis.ipynb
```

Run all cells via **Cell → Run All**, or step through with **Shift + Enter**.

### Option 2 — JupyterLab

```bash
pip install jupyterlab
jupyter lab Grocery_sales_analysis.ipynb
```

### Option 3 — VS Code

Open the `.ipynb` file in VS Code with the **Jupyter extension** installed and click **Run All**.

---

## 🔄 Pipeline Overview

```
Load CSVs → Normality Test → EDA (correlation, scatter matrix, boxplot, treemap, KDE) →
Split X/y → RobustScale → Train/Test Split →
[LR unscaled] → [KNN unscaled] → [LR scaled] → [KNN scaled] →
Residual Histograms → Forecast on Test Set → Consolidated Output Table
```

1. **Load** `Train.csv` and `Test.csv`
2. **Normality test** — D'Agostino's K² test on `GrocerySales`; confirm non-normal distribution
3. **EDA** — correlation heatmap (styled), scatter matrix, boxplot (via `pd.melt`), interactive Plotly treemap and OLS scatter, KDE density plot
4. **Prepare features** — `X = Day` (reshaped to column vector), `y = GrocerySales`, `X_pred = Test Day`
5. **Scale** — fit `RobustScaler` on full `X` → `X_scaled`; separately fit on `X_pred` → `X_pred_scaled`
6. **Train/test split** — 70/30 split (`random_state=42`) on both unscaled and scaled `X`
7. **Train and evaluate 4 models:**
   - Linear Regression on raw data
   - KNN Regressor (default `k=5`) on raw data
   - Linear Regression on scaled data
   - KNN Regressor (default `k=5`) on scaled data
8. **Residual histograms** — `sns.distplot` of `(test_y − predictions)` for all 4 models
9. **Forecast** — predict on `Test.csv` days using all 4 models; compile into a single results DataFrame
10. **Visualize forecasts** — 4 separate interactive Plotly scatter plots (one per model)

---

## 📊 Results

### Model Comparison on Validation Set (70/30 split of training data)

| Model | Scaling | MSE | MAE |
|---|---|---|---|
| **KNN Regressor** | **None** | **44,270.05** | **156.37** |
| KNN Regressor | RobustScaler | 45,194.10 | 157.37 |
| Linear Regression | None | 142,343.84 | 290.98 |
| Linear Regression | RobustScaler | 142,343.84 | 290.98 |

**KNN (unscaled) is the clear winner**, with MSE ~3.2× lower than Linear Regression and MAE roughly half. This is expected: daily grocery sales have a non-linear, locally-varying pattern that KNN's neighborhood averaging captures better than a global linear trend.

### Key Observations

**Linear Regression produces identical results scaled vs. unscaled** — this is mathematically correct. Scaling a single feature (Day) with a monotone transformation (RobustScaler) does not change the linear fit quality, only the coefficient scale. The notebook uses `RobustScaler.fit_transform(X_pred)` independently of `X`, which technically leaks future scaling information.

**KNN flat-lines on unscaled test predictions** — all forecast values are identical (8,993.078) because the unscaled future `Day` values (693+) fall outside the training range (1–692), so KNN always picks the same nearest neighbors from the edge of the training set. The scaled KNN forecast shows realistic variation because scaling maps future days into the interpolation range.

**Sales are non-normal** (p = 3.7×10⁻³³) — but this is not corrected in the pipeline. A log or Box-Cox transform on `GrocerySales` could reduce skew and improve both Linear Regression and KNN error metrics.

### Consolidated Forecast Output (first 2 rows)

| Day | LR (unscaled) | KNN (unscaled) | LR (scaled) | KNN (scaled) |
|---|---|---|---|---|
| 693 | 8,933.06 | 8,993.08 | 8,173.12 | 8,518.61 |
| 694 | 8,934.16 | 8,993.08 | 8,181.65 | 8,526.61 |

---

## ⚠️ Issues & Limitations

### Data Leakage in Scaled Test Predictions (Cell 20)

```python
# Current code — fits a NEW scaler on X_pred separately (leakage):
X_scaled     = rs.fit_transform(X)       # fit on training days
X_pred_scaled = rs.fit_transform(X_pred) # re-fits scaler on test days ← wrong

# Correct approach — transform X_pred using the scaler already fit on X:
X_scaled      = rs.fit_transform(X)
X_pred_scaled = rs.transform(X_pred)    # use same scaler, don't refit
```

Refitting on `X_pred` means the test days are scaled relative to themselves, not relative to the training distribution — causing the scaled LR predictions (starting at 8,173) to differ significantly from unscaled LR (starting at 8,933) when they should be equivalent.

### KNN Flat-Line on Unscaled Forecasts

As noted above, all 90 unscaled KNN forecast values are 8,993.078. This is a symptom of the extrapolation problem inherent to KNN — it cannot extrapolate beyond the training range. The scaled KNN predictions are more useful for this reason, despite the scaling leakage.

### Single Feature (`Day`) Limits Model Power

The only predictor is a sequential day counter. Neither model can capture seasonality, day-of-week effects, holidays, or promotions. A proper time-series approach (SARIMA, Prophet, or feature-engineered ML with calendar features) would significantly improve forecasts.

---

## 📂 File Structure

```
project/
│
├── Grocery_sales_analysis.ipynb    # Main notebook
├── Train.csv                        # Historical daily sales (days 1–692)
├── Test.csv                         # Future days for forecasting
└── README.md
```

---

## 📌 Notes

- `kurtosistest` from `scipy.stats` is imported but never called — it can be removed.
- `sns.distplot()` is deprecated in newer Seaborn versions (≥ 0.12); replace with `sns.histplot(..., kde=True)` to avoid warnings.
- Plotly interactive charts (`fig.show()`) require a running Jupyter server with a browser frontend — they will not render in static notebook viewers like nbviewer without an export step.
- A natural next step would be adding calendar features (day-of-week, week-of-year, month) extracted from a reference start date, then re-running both models — this alone typically cuts forecasting error by 30–50% on retail data.
