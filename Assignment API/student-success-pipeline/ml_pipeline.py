"""Sub-Objective 2: Machine Learning Pipeline (production-grade).

Model preparation, training (70/30 split) with hyper-parameter tuning and
cross-validation, evaluation with multiple metrics, MLOps logging and model
persistence for two algorithms: Random Forest and XGBoost.

Pre-processing (imputation) is embedded inside a scikit-learn ``Pipeline`` and
fitted on the training fold only, so there is no data leakage from the test set.
No feature scaling is applied because both models are tree-based and therefore
scale-invariant.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import joblib
import matplotlib
import mlflow
import mlflow.sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import config

_DEFAULT_LOGGER = logging.getLogger("ml_pipeline")


def _log(logger: Optional[logging.Logger], level: str, message: str) -> None:
    (logger or _DEFAULT_LOGGER).__getattribute__(level)(message)


# ---------------------------------------------------------------------------
# Weight-of-Evidence encoder (leakage-safe, fitted on the training fold only)
# ---------------------------------------------------------------------------
class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality nominal categoricals by Weight of Evidence.

    WoE is defined for a binary event. As the business goal is dropout
    prediction, the event is the Dropout class, and each category is mapped to::

        WoE = ln( P(category | Dropout) / P(category | not-Dropout) )

    with Laplace smoothing so rare categories stay finite. Because ``fit`` uses
    ``y`` it is applied inside the model pipeline and therefore fitted only on
    each training fold (no leakage). Unseen categories map to 0 (neutral).
    """

    def __init__(self, event_label: int = 0, smoothing: float = 0.5):
        self.event_label = event_label
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        event = (y == self.event_label).astype(int)
        n_event = int(event.sum())
        n_nonevent = int(len(event) - n_event)
        self.columns_ = list(X.columns)
        self.maps_ = {}
        for col in self.columns_:
            grouped = event.groupby(X[col])
            events = grouped.sum()
            nonevents = grouped.count() - events
            k = len(events)
            dist_event = (events + self.smoothing) / (n_event + self.smoothing * k)
            dist_nonevent = ((nonevents + self.smoothing)
                             / (n_nonevent + self.smoothing * k))
            self.maps_[col] = np.log(dist_event / dist_nonevent).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        return np.column_stack([
            X[col].map(self.maps_[col]).fillna(0.0).to_numpy()
            for col in self.columns_
        ])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"woe__{c}" for c in self.columns_], dtype=object)


# ---------------------------------------------------------------------------
# 2.1 Model preparation
# ---------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
):
    """Build the (raw) feature matrix X and encoded target y.

    Operates on the raw ingested data: de-duplicates, then separates features
    from the label. All imputation/scaling happens later inside the model
    pipeline to avoid leakage.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    # Binary framing: Dropout is the positive class (1); Enrolled/Graduate map to 0.
    y = (df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])
    _log(logger, "info", f"Prepared feature matrix with shape {X.shape} "
                         f"({len(df)} rows after de-duplication). Binary target: "
                         f"{int(y.sum())} Dropout / {int((y == 0).sum())} "
                         f"Not-Dropout.")
    return X, y


def _build_preprocessor(columns) -> ColumnTransformer:
    """Impute continuous, dummy-encode low-cardinality, WoE-encode nominals.

    Random Forest and XGBoost are tree-based, so continuous features are only
    imputed (no scaling needed). Categorical features are encoded explicitly:
    low-cardinality columns (Marital status + the binary flags) are one-hot
    (dummy) encoded, while high-cardinality nominals are Weight-of-Evidence
    encoded against the Dropout event. All fitting happens on the training fold
    only, so there is no leakage. This dataset has no missing values, so the
    imputer is a safety net (XGBoost also handles NaNs natively).
    """
    continuous = [c for c in config.CONTINUOUS_COLUMNS if c in columns]
    dummy = [c for c in config.DUMMY_COLUMNS if c in columns]
    woe = [c for c in config.WOE_COLUMNS if c in columns]
    return ColumnTransformer(
        transformers=[
            ("continuous", SimpleImputer(strategy="median"), continuous),
            ("dummy", OneHotEncoder(handle_unknown="ignore",
                                    sparse_output=False), dummy),
            ("woe", WeightOfEvidenceEncoder(
                event_label=config.WOE_EVENT_CLASS_INDEX), woe),
        ],
        remainder="drop",
    )


def _model_specs():
    """Estimators paired with their search spaces and iteration budgets.

    Each entry is ``(estimator, param_distribution, n_iter)``. XGBoost is given a
    much larger search space and iteration budget than Random Forest so it is
    tuned more thoroughly.
    """
    return {
        "RandomForest": (
            RandomForestClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
            config.TUNING_ITERATIONS,
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            {
                "model__n_estimators": [200, 300, 400, 600],
                "model__max_depth": [3, 4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
                "model__gamma": [0, 0.1, 0.3],
                "model__reg_lambda": [1.0, 2.0, 5.0],
            },
            config.XGB_TUNING_ITERATIONS,
        ),
    }


# ---------------------------------------------------------------------------
# 2.2 Model training (70/30 split + tuning + cross-validation)
# ---------------------------------------------------------------------------
def train_models(X, y, logger: Optional[logging.Logger] = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    _log(logger, "info", f"Train/test split: {len(X_train)} train / "
                         f"{len(X_test)} test rows (70/30, stratified).")

    preprocessor = _build_preprocessor(X.columns)
    trained = {}
    cv_results = {}

    for name, (estimator, param_dist, n_iter) in _model_specs().items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=3,
            scoring="f1",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        trained[name] = search.best_estimator_
        cv_results[name] = {
            "cv_f1": round(float(search.best_score_), 4),
            "n_iter": n_iter,
            "best_params": {k.replace("model__", ""): v
                            for k, v in search.best_params_.items()},
        }
        _log(logger, "info", f"Tuned {name} ({n_iter} candidates): cv_f1="
                             f"{cv_results[name]['cv_f1']} "
                             f"params={cv_results[name]['best_params']}")

    return trained, cv_results, X_test, y_test


# ---------------------------------------------------------------------------
# 2.3 / 2.4 Evaluation + MLOps metric logging & model persistence
# ---------------------------------------------------------------------------
def evaluate_models(models, cv_results, X_test, y_test,
                    logger: Optional[logging.Logger] = None) -> dict:
    all_metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(
                y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(
                y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(
                y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(
                y_test, y_proba[:, 1])), 4),
            "cv_f1": cv_results[name]["cv_f1"],
        }
        all_metrics[name] = metrics

        # MLOps: log each tracked metric individually.
        for metric_name, value in metrics.items():
            _log(logger, "info", f"[MLOps] {name} {metric_name} = {value}")

        # Per-class report for monitoring.
        report = classification_report(
            y_test, y_pred, target_names=config.BINARY_CLASSES,
            output_dict=True, zero_division=0)
        with open(config.REPORTS_DIR / f"classification_report_{name}.json",
                  "w") as fh:
            json.dump(report, fh, indent=2)

        _plot_confusion_matrix(y_test, y_pred, name)

    # --- Persist models (model registry) ---
    best = max(all_metrics, key=lambda m: all_metrics[m]["f1_score"])
    for name, model in models.items():
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
    joblib.dump(models[best], config.MODELS_DIR / "best_model.joblib")

    # --- Model card + timestamped metrics history (MLOps) ---
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": all_metrics,
        "best_model": best,
        "best_params": cv_results[best]["best_params"],
    }
    with open(config.REPORTS_DIR / "model_metrics.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    with open(config.REPORTS_DIR / "model_metrics_latest.json", "w") as fh:
        json.dump(record, fh, indent=2)
    with open(config.MODELS_DIR / "model_card.json", "w") as fh:
        json.dump(record, fh, indent=2)

    # --- MLflow experiment tracking (MLOps): one run per model ---
    _log_to_mlflow(all_metrics, cv_results, best, logger)

    _log(logger, "info", f"[MLOps] Best model by F1: {best} "
                         f"(f1={all_metrics[best]['f1_score']}). "
                         f"Saved to models/best_model.joblib.")
    return all_metrics


def _log_to_mlflow(all_metrics, cv_results, best,
                   logger: Optional[logging.Logger] = None) -> None:
    """Record each model's hyper-parameters, metrics and artifacts in MLflow."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                 artifact_location=config.MLFLOW_ARTIFACT_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    for name, metrics in all_metrics.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("is_best_model", str(name == best))
            mlflow.log_param("n_iter", cv_results[name].get("n_iter"))
            mlflow.log_params(cv_results[name]["best_params"])
            mlflow.log_metrics(metrics)
            report_path = config.REPORTS_DIR / f"classification_report_{name}.json"
            cm_path = config.PLOTS_DIR / f"confusion_matrix_{name}.png"
            for artifact in (report_path, cm_path):
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
    _log(logger, "info", f"[MLflow] Logged {len(all_metrics)} runs to "
                         f"{config.MLFLOW_TRACKING_URI} "
                         f"(experiment '{config.MLFLOW_EXPERIMENT}').")


def _plot_confusion_matrix(y_test, y_pred, name: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.BINARY_CLASSES,
                yticklabels=config.BINARY_CLASSES)
    plt.title(f"Confusion matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / f"confusion_matrix_{name}.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Inference helper (model serving)
# ---------------------------------------------------------------------------
def predict_from_saved(df: pd.DataFrame) -> np.ndarray:
    """Load the persisted best model and predict class labels for new rows."""
    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    features = df.drop(columns=[config.TARGET_COLUMN], errors="ignore")
    preds = model.predict(features)
    inverse = {idx: label for idx, label in enumerate(config.BINARY_CLASSES)}
    return np.array([inverse[int(p)] for p in preds])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    import data_pipeline as dp
    # Route training through the imported module so the custom
    # WeightOfEvidenceEncoder pickles under "ml_pipeline" (not "__main__"),
    # keeping the saved model loadable from any context.
    import ml_pipeline as mlp

    _df = dp.ingest_data()
    dp.validate_data(_df)
    X, y = mlp.prepare_features(_df)
    _models, _cv, X_test, y_test = mlp.train_models(X, y)
    results = mlp.evaluate_models(_models, _cv, X_test, y_test)
    print(json.dumps(results, indent=2))
