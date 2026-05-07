# 🚗 Used Car Auction Market Intelligence Pipeline

> **Pricing Inconsistency & Deal Scoring Analysis — 2014–2015 Auction Data**

A full end-to-end data science project covering data ingestion, cleaning, feature engineering, exploratory analysis, machine learning modeling, and a Power BI dashboard — all built on 558,837 real-world used car auction records.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
- [Machine Learning Models](#machine-learning-models)
- [Dashboard](#dashboard)
- [Key Results](#key-results)
- [Setup & Usage](#setup--usage)
- [Dependencies](#dependencies)

---

## Project Overview

**Objective:** Analyze 524,666 cleaned auction records to identify pricing inefficiencies, quantify depreciation drivers, and flag undervalued or overpriced inventory against MMR (Manheim Market Report) benchmarks.

**Core Questions:**
- Which vehicles are selling significantly below or above market value?
- How do vehicle age, mileage, and condition drive price deviation from MMR?
- Can we predict auction selling price with high accuracy?

---

## Repository Structure

```
├── data.csv                    # Raw auction dataset (558,837 records)
├── cleaned_auction_data.csv    # Cleaned & engineered dataset (524,666 records)
├── notebook.ipynb              # Main EDA & pipeline notebook
├── model.ipynb                 # ML modeling notebook (Ridge + XGBoost)
├── best_model.pkl              # Serialized best model (XGBoost pipeline)
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Raw records | 558,837 |
| Cleaned records | 524,666 |
| Features (raw) | 17 columns |
| Features (engineered) | 20 columns |
| Date range | 2014 – 2015 |
| Avg. Selling Price | $12,826 |
| Median Selling Price | $12,000 |
| Avg. Vehicle Age | 4.8 years |
| Avg. Odometer | 66,397 miles |
| % Sold Above MMR | 46.5% |

**Raw Columns:** `Id`, `Year`, `Make`, `Model`, `Trim`, `Body`, `Transmission`, `VIN`, `State`, `ConditionValue`, `Odometer`, `Color`, `Interior`, `Seller`, `MMR`, `SellingPrice`, `SaleDate`

**Engineered Columns Added:** `VehicleAge`, `AgeGroup`, `MileageGroup`, `LogSellingPrice`, `PriceVsMMR`, `PctDiff`, `MarketSignal`, `SaleYear`, `SaleMonth`

---

## Pipeline Stages

The pipeline follows a structured six-stage data science lifecycle (`notebook.ipynb`):

### Stage 01 — Environment Setup
Initializes the analytics stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scipy`. Applies consistent currency formatting across all visualizations.

### Stage 02 — Data Collection & Initial Audit
Loads the raw 558,837-row dataset and audits schema integrity, column types, and missing value distribution. Notable gaps include `Transmission` (11.69% missing) and `Body` (2.36% missing).

### Stage 03 — Data Cleaning & Normalization
- **Numerical nulls:** Filled with column median; `ConditionValue` uses group-level median (Make/Model/Year) before falling back to global median.
- **Categorical normalization:** Title-casing for `Make`, `Model`, `Body`, `Color`, `Interior`; uppercasing for `State`.
- **Placeholder handling:** Em-dash (`—`) and dash (`-`) values replaced with `NaN`, then filled with `"Unknown"`.
- **Transmission imputation:** Mode-based fill using (Make, Model, Year) group-level lookup.
- **Outlier removal:** IQR-based filtering on `SellingPrice` and `Odometer`, removing extreme tails. Final cleaned dataset: **524,666 records**.

### Stage 04 — Feature Engineering & Deal Scoring
| Feature | Description |
|---|---|
| `VehicleAge` | `SaleYear − Year` |
| `AgeGroup` | Bucketed: `New (0–2)`, `Recent (3–5)`, `Mid-Age (6–10)`, `Older (11+)` |
| `MileageGroup` | Bucketed: `Low (<30k)`, `Medium (30–70k)`, `High (70–120k)`, `Very High (>120k)` |
| `LogSellingPrice` | Log-transformed selling price for normalization |
| `PriceVsMMR` | `SellingPrice − MMR` (raw dollar deviation) |
| `PctDiff` | `(SellingPrice − MMR) / MMR × 100` (percentage deviation) |
| `MarketSignal` | `Undervalued` (<−5%), `Fair Value` (±5%), `Overpriced` (>+5%) |

### Stage 05 — Data Visualization
Key visual analyses produced:
- **Price distribution** by `AgeGroup` and `MileageGroup` (box plots)
- **Depreciation curve** — median selling price vs. vehicle age
- **Condition impact** — `ConditionValue` vs. `SellingPrice` scatter with regression line
- **Market Signal segmentation** — proportion of Undervalued / Fair Value / Overpriced vehicles
- **Monthly median price trend** — time-series over the 2014–2015 auction period

### Stage 06 — Interpretation & Export
Final cleaned and engineered dataset saved to `cleaned_auction_data.csv` (20 columns). Summary statistics printed for production handoff.

---

## Machine Learning Models

Implemented in `model.ipynb`. Both models use a shared preprocessing pipeline with `OneHotEncoder` for categorical features and pass-through for numerical features.

**Features used for modeling:**

| Type | Columns |
|---|---|
| Categorical | `Make`, `Model`, `Body`, `Transmission`, `State`, `AgeGroup`, `MileageGroup`, `MarketSignal` |
| Numerical | `Year`, `VehicleAge`, `Odometer`, `MMR`, `ConditionValue`, `SaleYear`, `SaleMonth` |

**Target:** `SellingPrice`  
**Split:** 80% train / 20% test (random state 42)

### Model Comparison

| Model | MAE | R² |
|---|---|---|
| Ridge Regression (α=1.0) | $1,064.60 | 0.9528 |
| **XGBoost** (300 trees, lr=0.05, depth=6) | **$592.05** | **0.9862** |

XGBoost significantly outperforms Ridge on both metrics and was saved as `best_model.pkl`.

**XGBoost Hyperparameters:**
```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Loading the saved model:**
```python
import joblib
model = joblib.load("best_model.pkl")
predictions = model.predict(X_new)
```

---

## Dashboard

A **Power BI dashboard** was built on top of `cleaned_auction_data.csv` to provide interactive market intelligence for stakeholders.

<img width="1143" height="772" alt="Power BI Dashboard" src="https://github.com/user-attachments/assets/954ffaa1-669d-4099-a834-bea1c69dca44" />

**Dashboard highlights:**
- Market Signal breakdown (Undervalued / Fair Value / Overpriced) across Makes and States
- Price vs. MMR deviation heatmaps by region and vehicle category
- Depreciation curves filtered by AgeGroup and MileageGroup
- Monthly auction price trend with volume overlay
- Top undervalued inventory list — vehicles with the highest negative `PctDiff`

> The dashboard consumes `cleaned_auction_data.csv` directly. Refresh the Power BI data source after re-running `notebook.ipynb` to reflect updated data.

---

## Key Results

- **XGBoost achieved a MAE of $592** on held-out test data, predicting auction prices within ~4.6% of the median selling price on average.
- **46.5% of vehicles sold above their MMR benchmark**, indicating a consistently competitive auction environment.
- `MMR` is the single strongest predictor of `SellingPrice` — but vehicle age, mileage group, and condition together explain the residual deviation.
- Undervalued vehicles are disproportionately concentrated in the **High Mileage + Mid-Age** segment, representing the clearest arbitrage opportunity.

---

## Setup & Usage

### 1. Clone the repository and install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost joblib
```

### 2. Run the EDA pipeline

Open and run all cells in `notebook.ipynb`. This will:
- Load `data.csv`
- Clean, engineer, and export `cleaned_auction_data.csv`

### 3. Train the models

Open and run all cells in `model.ipynb`. This will:
- Load `cleaned_auction_data.csv`
- Train Ridge and XGBoost models
- Save the best model to `best_model.pkl`

### 4. Open the Dashboard

Connect Power BI to the generated `cleaned_auction_data.csv` to explore the interactive dashboard.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scipy` | Statistical analysis |
| `scikit-learn` | Preprocessing, Ridge, pipelines, metrics |
| `xgboost` | Gradient boosting model |
| `joblib` | Model serialization |
| Power BI Desktop | Interactive dashboard |

---

*Dataset covers U.S. wholesale auction records from 2014–2015. MMR benchmarks sourced from Manheim Market Report valuations embedded in the original data.*
