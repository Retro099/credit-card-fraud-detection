# Credit Card Fraud Detection

**One line:** an XGBoost model that catches 84% of fraud on a dataset where only 0.172% of transactions are fraud, with SHAP explanations, a Streamlit demo, Docker and tests.

[![CI](https://github.com/Retro099/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Retro099/credit-card-fraud-detection/actions/workflows/ci.yml)

**Live demo:** https://ml-projects-credit-card-fraud-detection.streamlit.app/ (deployed from the original `ML-Projects` monorepo)

## Results

Test split: 56,962 transactions, 98 fraud. Threshold 0.5. Source: committed outputs of `notebooks/Credit_Card_Fraud.ipynb` (single run, not re-trained for this repo).

| Model | Recall (fraud) | Precision (fraud) | PR-AUC | Notes |
|---|---|---|---|---|
| Logistic regression (balanced) | 0.92 | 0.06 | n/a | catches fraud but flags far too many normal ones |
| Random forest (balanced_subsample) | 0.81 | 0.79 | 0.82 | |
| **XGBoost + scale_pos_weight (shipped)** | **0.84** | **0.88** | **0.88** | exact: 0.8367 / 0.8817 / 0.8787 |
| LightGBM (balanced) | 0.84 | 0.84 | 0.87 | close second |

Top SHAP drivers: **V14** and **V17** (strongly negative values push the score toward fraud).

## Quickstart

```bash
pip install -r requirements-dev.txt
pytest -q                      # 6 tests, no dataset needed
python predict.py              # CLI prediction on a dummy row
streamlit run app.py           # local demo
docker compose up --build      # same, in Docker (http://localhost:8501)
```

### Dataset (not committed)
Kaggle "Credit Card Fraud Detection" (ULB, 284,807 transactions): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Download `creditcard.csv` into `data/` (git-ignored) and run the notebook (Colab works) to retrain.

## How it works
1. Features: `Time`, `Amount` and the anonymised PCA components `V1`–`V28`.
2. Stratified train/test split; imbalance handled with class weights (`scale_pos_weight`) rather than oversampling.
3. Compared logistic regression, resampling variants, random forest, XGBoost and LightGBM on recall, precision and PR-AUC (not accuracy: a model that says "never fraud" is 99.83% accurate).
4. SHAP explains which features push each prediction toward fraud.
5. The chosen pipeline is saved as `artifacts/fraud_model_v1.joblib` (0.5 MB) and served by `predict.py` and `app.py`.

## Limitations
- Single train/test split; no cross-validation or confidence intervals.
- Threshold fixed at 0.5; it was not tuned on a separate validation split.
- V1–V28 are anonymised PCA features, so SHAP shows *which* component matters, not a business meaning.
- Public 2013 European dataset; real systems need drift monitoring and retraining.
- Model pickle needs `scikit-learn==1.6.1` (pinned).

## Project history
Split out of [Retro099/ML-Projects](https://github.com/Retro099/ML-Projects) (archived) on 2026-10-12.

License: MIT
