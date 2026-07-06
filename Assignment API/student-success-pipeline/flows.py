"""Prefect flows for the cloud-native pipeline (DataOps + MLOps).

Defines tasks and flows that wrap the data and ML pipelines. Prefect captures
every log line below and displays them on the dashboard, satisfying the
"log all activity details and display them on a Cloud dashboard" requirement.

Run locally:
    python flows.py                 # one-off run of the full pipeline

Schedule every 2 minutes (DataOps):
    python deploy.py                # serves the data pipeline on a 120s interval
"""
from __future__ import annotations

import os

# Allow running flows standalone (no separate server) via a temporary server.
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")

from prefect import flow, task
from prefect.logging import get_run_logger

import config
import data_pipeline as dp
import ml_pipeline as mlp


# ---------------------------------------------------------------------------
# Data pipeline tasks (Sub-Objective 1)
# ---------------------------------------------------------------------------
@task(name="ingest-data", retries=2, retry_delay_seconds=5)
def ingest_task():
    logger = get_run_logger()
    return dp.ingest_data(logger=logger)


@task(name="validate-data")
def validate_task(df):
    logger = get_run_logger()
    return dp.validate_data(df, logger=logger)


@task(name="preprocess-data")
def preprocess_task(df):
    logger = get_run_logger()
    return dp.preprocess_data(df, logger=logger)


@task(name="exploratory-data-analysis")
def eda_task(df):
    logger = get_run_logger()
    return dp.run_eda(df, logger=logger)


# ---------------------------------------------------------------------------
# ML pipeline tasks (Sub-Objective 2)
# ---------------------------------------------------------------------------
@task(name="prepare-features")
def prepare_task(df):
    logger = get_run_logger()
    return mlp.prepare_features(df, logger=logger)


@task(name="train-models")
def train_task(X, y):
    logger = get_run_logger()
    return mlp.train_models(X, y, logger=logger)


@task(name="evaluate-models")
def evaluate_task(models, cv_results, X_test, y_test):
    logger = get_run_logger()
    return mlp.evaluate_models(models, cv_results, X_test, y_test, logger=logger)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
@flow(name="data-pipeline")
def data_pipeline_flow():
    """DataOps: ingestion -> validation -> preprocessing -> EDA, fully logged."""
    logger = get_run_logger()
    logger.info("Starting data pipeline flow.")
    df_raw = ingest_task()
    validate_task(df_raw)
    df_clean = preprocess_task(df_raw)
    eda_report = eda_task(df_clean)
    logger.info("Data pipeline flow completed successfully.")
    return eda_report


@flow(name="ml-pipeline")
def ml_pipeline_flow():
    """MLOps: prepare -> train (RF + XGBoost) -> evaluate with 4+ metrics."""
    logger = get_run_logger()
    logger.info("Starting ML pipeline flow.")
    df_raw = ingest_task()
    validate_task(df_raw)
    X, y = prepare_task(df_raw)
    models, cv_results, X_test, y_test = train_task(X, y)
    metrics = evaluate_task(models, cv_results, X_test, y_test)
    logger.info("ML pipeline flow completed successfully.")
    return metrics


@flow(name="student-success-pipeline")
def full_pipeline_flow():
    """End-to-end orchestration of the data and ML pipelines."""
    logger = get_run_logger()
    logger.info("Running full student-success pipeline (data + ML).")
    df_raw = ingest_task()
    validate_task(df_raw)
    df_clean = preprocess_task(df_raw)
    eda_task(df_clean)
    X, y = prepare_task(df_raw)
    models, cv_results, X_test, y_test = train_task(X, y)
    metrics = evaluate_task(models, cv_results, X_test, y_test)
    logger.info("Full pipeline finished.")
    return metrics


if __name__ == "__main__":
    full_pipeline_flow()
