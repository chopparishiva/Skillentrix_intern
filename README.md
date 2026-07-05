# Telco Customer Churn Prediction

Predicting customer churn for a telecommunications company using classical ML and gradient boosting models, with a full evaluation and explainability layer built in.

## Overview

This project predicts which telecom customers are likely to churn, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 19 features covering demographics, account info, and subscribed services).

It's organized as both:
- An **exploratory notebook** (`notebooks/`) for EDA, step-by-step model building, and visual walkthroughs.
- A **reusable Python pipeline** (`src/`) for training, evaluating, explaining, and scoring new data from the command line — the part you'd actually deploy or schedule.

## Project Structure

```
.
├── data/
│   └── telco_customer_churn.csv       # raw dataset
├── notebooks/
│   └── telco_customer_churn_eda.ipynb # EDA + model walkthrough (human-readable narrative)
├── src/
│   ├── config.py                      # paths, random seed, feature groups (single source of truth)
│   ├── data_preprocessing.py          # cleaning, encoding, scaling, train/test split
│   ├── train.py                       # trains & persists all 5 models
│   ├── evaluate.py                    # classification metrics, comparison table, ROC curves
│   ├── explain.py                     # SHAP feature-importance plots
│   └── predict.py                     # CLI to score new customer records
├── tests/
│   └── test_data_preprocessing.py     # unit tests for the preprocessing pipeline
├── models/                            # trained model artifacts (generated, gitignored)
├── reports/figures/                   # generated plots (generated, gitignored)
├── requirements.txt                   # pinned runtime dependencies
├── requirements-dev.txt               # + pytest, for running tests
└── LICENSE
```

## Models

Five classifiers are trained and compared, all seeded with the same random state (`44`, in `src/config.py`) for reproducibility:

- Logistic Regression
- Decision Tree
- LightGBM
- Random Forest
- XGBoost

## Evaluation

Beyond accuracy/precision/recall/F1, models are compared with **ROC-AUC**, which is the more meaningful metric here given the ~27% churn rate (i.e., an imbalanced target where accuracy alone is misleading). `src/evaluate.py` produces:
- A single comparison table across all five models (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`), sorted by ROC-AUC.
- An overlaid ROC curve plot for all models.

Run it yourself to generate current numbers — see [Usage](#usage). Results will vary slightly depending on your installed library versions, so no specific scores are hardcoded here.

## Explainability

`src/explain.py` uses **SHAP (SHapley Additive exPlanations)** on the trained XGBoost model to show which features drive churn predictions and in which direction, rather than treating the model as a black box. It produces both a global feature-importance bar chart and a per-customer beeswarm summary plot, saved to `reports/figures/`.

## Installation

```bash
git clone <your-repo-url>
cd Telco-Customer-Churn
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For running tests as well:
```bash
pip install -r requirements-dev.txt
```

## Usage

Train all five models (saves to `models/`):
```bash
python -m src.train
```

Evaluate them (prints per-model classification reports, a comparison table, and saves an ROC curve plot to `reports/figures/roc_curves.png`):
```bash
python -m src.evaluate
```

Generate SHAP explainability plots for the XGBoost model (saved to `reports/figures/`):
```bash
python -m src.explain
```

Score a new batch of customers (CSV with the same raw columns as `data/telco_customer_churn.csv`, minus `Churn`):
```bash
python -m src.predict --input path/to/new_customers.csv --model xgboost
```

Run the test suite:
```bash
pytest tests/
```

Alternatively, open `notebooks/telco_customer_churn_eda.ipynb` for the full narrative walkthrough — EDA, feature-risk analysis, and the same models built step by step with inline explanations.

## Reproducibility notes

- All models share one random seed (`src/config.py: RANDOM_STATE = 44`).
- The train/test split is **stratified** on the target, so class balance is preserved in both sets.
- `requirements.txt` pins dependency versions; SHAP's API has changed across versions in ways that affect output shape, so version drift is a real risk if you unpin it (`src/explain.py` handles the known shape variations defensively, but hasn't been tested against every SHAP release).
- I was not able to execute `train.py`/`evaluate.py`/`explain.py` end-to-end with LightGBM, XGBoost, and SHAP installed together in the environment used to build this repo (no network access there). The `src/data_preprocessing.py` pipeline and the scikit-learn-only models (Logistic Regression, Decision Tree, Random Forest) were verified to run correctly, including the comparison table and ROC plotting logic. Run `python -m src.train` yourself as a first step after cloning to confirm the boosting models and SHAP work in your environment.

## Known limitations / not yet done

Being upfront about what this repo doesn't do, so it doesn't oversell itself:
- No hyperparameter tuning (all models use fixed or default parameters) — results reflect baseline model capacity, not tuned performance.
- No class-imbalance handling (e.g., class weighting or resampling) beyond stratified splitting.
- No CI pipeline (e.g., GitHub Actions running `pytest` on push) — the tests exist but aren't automated yet.
- `src/predict.py` assumes new data has the same categories the label encoders were fit on; unseen category values will raise an error rather than degrading gracefully.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — customer demographics, account details, subscribed services, and churn status for a telecom provider.

## License

[MIT](LICENSE)

## Author

George Youhana
