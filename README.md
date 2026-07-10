# Predicting Student Dropout & Academic Success — Cloud-Native ML Application

API-driven Cloud Native Solutions (S2-25_AIMLCZG549) — Assignment I.

A cloud-native Data Science / Machine Learning application built with **Prefect**
that ingests the UCI *Predict Students' Dropout and Academic Success* dataset,
runs an automated data pipeline and ML pipeline, schedules them every 2 minutes,
logs all activity to a cloud dashboard, and exposes application details through
Prefect's built-in REST API.

## Business problem

Higher-education institutions lose revenue and reputation when students drop out.
Using academic, demographic and socio-economic attributes recorded at enrollment
and after the first two semesters, we predict whether a student will **Dropout**
or **not** (i.e. remain enrolled or graduate) — enabling early intervention.

- **Dataset:** [UCI ML Repository #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Records:** 4,424 · **Features:** 36 · **Target:** binary (Dropout vs Not-Dropout, i.e. Enrolled/Graduate collapsed)
- Saved locally as `../data.csv` (semicolon-delimited).

## Mapping to the assignment

| Assignment requirement | Where it is implemented |
| --- | --- |
| 1.2 Data ingestion | `data_pipeline.ingest_data` |
| Data validation / quality gate | `data_pipeline.validate_data` (schema, duplicates, imbalance, outliers) |
| 1.3 Pre-processing (summary stats, missing values, typed imputation, dtypes, normalization) | `data_pipeline.preprocess_data` |
| 1.4 EDA (correlation, binning, encoding, feature importance, viz) | `data_pipeline.run_eda` |
| 1.5 DataOps (automated, scheduled every 2 min, logged, dashboard) | `flows.py` + `deploy.py` |
| 2.1–2.3 Model prep / training (70/30, tuning, CV) / evaluation | `ml_pipeline.py` |
| 2.4 MLOps (log ≥ 4 metrics + ROC-AUC, model registry) | `ml_pipeline.evaluate_models` |
| 3.1–3.2 API access (flows, deployments, runs, work pools) | `api_access.py` |

## Project layout

```
student-success-pipeline/
├── config.py            # paths, dataset & ML configuration
├── data_pipeline.py     # ingestion, preprocessing, EDA  (Sub-Objective 1)
├── ml_pipeline.py       # RandomForest + XGBoost, metrics (Sub-Objective 2)
├── flows.py             # Prefect tasks & flows (DataOps/MLOps)
├── deploy.py            # 2-minute scheduled deployments
├── api_access.py        # Prefect built-in REST API access (Sub-Objective 3)
├── requirements.txt
└── outputs/             # generated artifacts, reports, plots, logs
```

## Setup

```bash
cd "student-success-pipeline"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### 1. Run pipelines once (local)

```bash
python data_pipeline.py     # data pipeline only
python ml_pipeline.py       # data + ML pipeline, prints metrics
python flows.py             # full Prefect flow (data + ML)
```

Artifacts appear under `outputs/`:
- `outputs/reports/` — summary stats, correlation matrix, feature importance, metrics
- `outputs/plots/` — distributions, heatmap, feature importance, confusion matrices
- `outputs/artifacts/` — preprocessed dataset

### 2. DataOps — schedule every 2 minutes + dashboard

```bash
# Terminal A — start the Prefect server & dashboard (http://127.0.0.1:4200)
prefect server start

# Terminal B — point the client at the server and serve the deployments
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python deploy.py
```

Open <http://127.0.0.1:4200> to watch flow runs trigger every 2 minutes with
full logs (the **Cloud dashboard** for DataOps/MLOps monitoring). To deploy to
**Prefect Cloud** instead, run `prefect cloud login` and skip `prefect server start`.

### 3. API access — display application details

With the server running and at least one deployment served:

```bash
python api_access.py
```

This calls Prefect's built-in REST API and prints ≥ 4 application details:
API health, version, registered flows, deployments (with schedules/tags),
recent flow runs and their states, and work pools.

## Models & metrics

Both algorithms are wrapped in a scikit-learn `Pipeline` fitted on the training
fold only, so there is **no data leakage**. Categorical features are encoded
explicitly: low-cardinality columns (`Marital status` + the binary flags) are
dummy/one-hot encoded, while high-cardinality nominals (Course, Nacionality,
parents' qualifications/occupations, etc.) are **Weight-of-Evidence** encoded
against the Dropout event. No feature scaling is applied because Random Forest
and XGBoost are tree-based and scale-invariant (normalization is still
demonstrated in the data pipeline as required by step 1.3). Each model is tuned
with `RandomizedSearchCV` (stratified cross-validation); **XGBoost gets a deeper,
dedicated search** (8 hyper-parameters, 30 candidates) versus Random Forest's
smaller grid. Models are evaluated on the held-out 30% test set with six
metrics: accuracy, precision, recall, F1 and ROC-AUC for the Dropout class, plus
the cross-validated F1. Every run's hyper-parameters, metrics and artifacts are
tracked in **MLflow** (SQLite backend at `outputs/mlflow.db`, artifacts in
`outputs/mlartifacts`) in addition to the
appended `outputs/reports/model_metrics.jsonl` history. Metrics
are appended to `outputs/reports/model_metrics.jsonl` on every run so they can be
monitored over time (MLOps). Trained pipelines are persisted to `outputs/models/`
(`best_model.joblib` + `model_card.json`) as a lightweight model registry; use
`ml_pipeline.predict_from_saved(df)` for inference.

Inspect the tracked experiments with the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

| Model | Notes |
| --- | --- |
| Random Forest | tuned, `class_weight="balanced"` for imbalance |
| XGBoost | tuned, `multi:softprob` |
