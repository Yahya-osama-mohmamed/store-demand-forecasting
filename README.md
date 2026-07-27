# 📦 Store Item Demand Forecasting

[![CI Pipeline](https://github.com/Yahya-osama-mohmamed/store-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Yahya-osama-mohmamed/store-demand-forecasting/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project to forecast daily item-level demand across retail stores. This repository contains a production-ready ML pipeline, a FastAPI REST service, a Streamlit dashboard, and complete deployment configurations.

![Dashboard usage](docs/demand_ui.gif)

*Live dashboard: pick any store & item → 90-day forecast over historical context + SHAP explanation.*

![Model metrics](docs/model_metrics.png)


## 🎯 Business Problem

Demand forecasting drives every downstream retail decision: inventory purchasing, staffing, logistics, and promotions. Under-forecasting causes stockouts and lost sales; over-forecasting ties up capital in inventory. This project predicts **daily sales for 50 items across 10 stores** (500 concurrent time series) with a 3-month forecast horizon.

**Dataset:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle)
(913,000 daily records: 10 stores × 50 items × 5 years, 2013–2017). Downloaded automatically from a public mirror — no Kaggle credentials needed.

---

## 🛠️ Tech Stack

- **Data Science Core:** pandas, NumPy, scikit-learn
- **Machine Learning Models:** LightGBM, XGBoost, HistGradientBoosting
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Experiment Tracking:** MLflow
- **API & Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend / Dashboard:** Streamlit, Plotly
- **DevOps / CI-CD:** Docker, Docker Compose, GitHub Actions, Render
- **Testing:** Pytest

---

## 🚀 Quick Start

### 1. Local Setup (Without Docker)

```bash
git clone https://github.com/Yahya-osama-mohmamed/store-demand-forecasting.git
cd store-demand-forecasting
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
Downloads the dataset, cleans it, engineers features, trains and tunes all three models, generates SHAP explanations, tracks experiments in MLflow, and saves the final `.joblib` pipeline.
```bash
python main.py
```

### 3. Start the Applications
**FastAPI Backend (Port 8000):**
```bash
uvicorn app.api:app --reload
```
Swagger documentation available at: http://localhost:8000/docs

**Streamlit Dashboard (Port 8501):**
```bash
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Setup

```bash
# Start both API and Streamlit containers
docker-compose up --build -d

# Check logs
docker-compose logs -f
```
The pipeline (`main.py`) must be run at least once locally to generate the `models/` directory before starting the containers.

---

## 📂 Project Structure

```
.
├── app/                    # Deployment Layer
│   ├── api.py              # FastAPI application
│   ├── schemas.py          # Pydantic models
│   └── streamlit_app.py    # Streamlit dashboard
├── data/                   # Data directory (ignored in git)
│   ├── raw/                # Original downloaded dataset
│   └── processed/          # Cleaned & split data
├── dashboard/              # Power BI dashboard (PBIP project format)
├── figures/                # EDA and SHAP visualizations
├── models/                 # Saved joblib models and pipelines
├── mlruns/                 # MLflow tracking store
├── notebooks/              # Jupyter notebooks for exploration
├── reports/                # Executive summaries and comparison tables
├── src/                    # Core ML Source Code
│   ├── config.py           # Constants, paths, split boundaries
│   ├── data_loader.py      # Download and initial profiling
│   ├── preprocessing.py    # Cleaning, chronological split, encoding
│   ├── feature_engineering.py # Calendar features + learned aggregates
│   ├── feature_selection.py   # MI and Random Forest importance
│   ├── model_training.py   # RandomizedSearchCV + TimeSeriesSplit tuning
│   ├── model_evaluation.py # SMAPE/RMSE/MAE metrics and plots
│   ├── explainability.py   # SHAP analysis
│   └── logger.py           # Structured JSON logging
├── tests/                  # Pytest unit tests
├── Dockerfile              # Multi-purpose Dockerfile
├── docker-compose.yml      # Container orchestration
├── main.py                 # Pipeline execution entry point
├── render.yaml             # Render cloud deployment config
└── requirements.txt        # Python dependencies
```

---

## 📊 Model Performance

After running the pipeline, check `reports/model_comparison.csv` for detailed metrics. The primary metric is **SMAPE** (the competition metric — lower is better); leading public-leaderboard solutions score ~12.5–14.

### Methodology Notes

- **Chronological splits** — train ≤ 2017-06, validate on 2017-07..09, test on
  2017-10..12 (matching the competition's 3-month horizon). Random splits would
  leak the future into training.
- **Model selection** uses the validation window; the test window is reserved
  strictly for the final unbiased estimate.
- **Feature engineering lives inside the sklearn Pipeline** (`FeatureEngineer`
  step): calendar/cyclical features plus per-entity demand aggregates
  (store-item mean, store-item day-of-week mean, seasonal means) learned from
  training folds only. The saved `models/final_pipeline.joblib` accepts raw
  `(date, store, item)` records — no manual feature engineering at serving time,
  and any future date works.
- **Hyperparameter tuning** uses `RandomizedSearchCV` with `TimeSeriesSplit`, so
  every CV fold validates on data strictly after its training window.
- Feature-importance analysis (mutual information + random forest) is reported
  in `figures/feature_selection_*.png`; models train on the full feature set.

### Why these three models?

Histogram-based gradient boosting dominates this competition's leaderboard —
deep learning generally underperforms GBMs on tabular data of this size:
1. **LightGBM** — leaf-wise growth, typically the strongest here.
2. **XGBoost** — depth-wise with strong regularization; a robust second opinion.
3. **HistGradientBoosting** — scikit-learn's native LightGBM-style GBM;
   equally competitive with zero extra dependencies.

---

## 📊 Power BI Dashboard — Demand Planning Control Tower

A four-page interactive Power BI dashboard built on the model's forecast output:

![Power BI dashboard usage](docs/dashboard.gif)

*Live usage: Executive Overview (5-year trend + growth KPIs) → Seasonality &
Planning (store × month planning grid, weekday/weekend rhythm) → Forecast
Explorer (pick any store/item → 90-day plan with history beside it) → Forecast
Accuracy (held-out quarter: actual vs forecast, error by store, champion model
card).*

Open `dashboard/DemandPlanning/DemandPlanning.pbip` with Power BI Desktop
(PBIP/PBIR project format — enable *Power BI Project files* in Preview
features).

---

## ☁️ Deployment

**AWS serverless inference** (S3 + Lambda + API Gateway, $0 idle cost): see
[`aws/README.md`](aws/README.md). The trained pipeline is exported to
dependency-light primitives (parity-verified to 1e-6 against the sklearn
pipeline — `aws/parity_check.py`) and served from a ~40MB Lambda package
with least-privilege IAM. Lambda was chosen over SageMaker deliberately:
no hourly-billing endpoint risk, permanent 1M-request free tier.

Also configured for **Render** via `render.yaml` (FastAPI service + Streamlit
dashboard). CI runs all `pytest` suites via GitHub Actions.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🖼️ Output Gallery

| | |
|---|---|
| ![Forecast vs actual](figures/forecast_vs_actual_test.png) | ![Demand drivers](figures/shap_bar.png) |
| ![Seasonality](figures/eda_monthly_seasonality.png) | ![Weekly rhythm](figures/eda_dayofweek.png) |
| ![Daily trend](figures/eda_daily_sales.png) | ![Store x month](figures/eda_store_month_heatmap.png) |
| ![Model comparison](figures/model_comparison_bar.png) | ![Predicted vs actual](figures/pred_vs_actual_xgboost.png) |

**AWS deployment evidence** (live before teardown — see [aws/DEPLOYMENT_LOG.md](aws/DEPLOYMENT_LOG.md)):

![AWS evidence](aws/screenshots/aws_deployment_evidence.png)
