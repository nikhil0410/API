"""Sub-Objective 1: Data Pipeline.

Implements data ingestion, pre-processing and exploratory data analysis (EDA)
for the UCI "Predict Students' Dropout and Academic Success" dataset.

Every function accepts an optional ``logger`` so the same code can be reused
directly inside Prefect tasks (DataOps) and from notebooks / the CLI.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend so it works on cloud workers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

import config

_DEFAULT_LOGGER = logging.getLogger("data_pipeline")


def _log(logger: Optional[logging.Logger], level: str, message: str) -> None:
    (logger or _DEFAULT_LOGGER).__getattribute__(level)(message)


# ---------------------------------------------------------------------------
# 1.2 Data Ingestion
# ---------------------------------------------------------------------------
def ingest_data(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read the raw CSV downloaded from the UCI repository."""
    df = pd.read_csv(config.RAW_DATA_PATH, sep=config.CSV_SEPARATOR)
    # Some column headers in the raw file contain trailing tabs/spaces.
    df.columns = [c.strip() for c in df.columns]
    _log(logger, "info", f"Ingested {len(df)} rows and {df.shape[1]} columns "
                         f"from {config.RAW_DATA_PATH.name}")
    return df


# ---------------------------------------------------------------------------
# Data validation / quality gate (DataOps maturity)
# ---------------------------------------------------------------------------
def validate_data(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> dict:
    """Run schema and quality checks before the data is used downstream.

    Produces a data-quality report and raises ``ValueError`` on critical
    failures (missing target, too few rows) so a scheduled run fails loudly.
    """
    report: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}
    issues: list[str] = []

    # --- Schema checks ---
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Critical: target column "
                         f"'{config.TARGET_COLUMN}' is missing.")
    n_features = df.shape[1] - 1
    report["n_rows"] = int(len(df))
    report["n_features"] = int(n_features)
    if len(df) < config.MIN_EXPECTED_ROWS:
        raise ValueError(f"Critical: only {len(df)} rows "
                         f"(< {config.MIN_EXPECTED_ROWS}).")
    if n_features != config.EXPECTED_FEATURE_COUNT:
        issues.append(f"Expected {config.EXPECTED_FEATURE_COUNT} features, "
                      f"found {n_features}.")

    # --- Missing values ---
    missing = df.isnull().sum()
    report["columns_with_missing"] = {
        k: int(v) for k, v in missing[missing > 0].items()
    }
    report["total_missing"] = int(missing.sum())

    # --- Duplicates ---
    report["duplicate_rows"] = int(df.duplicated().sum())
    if report["duplicate_rows"]:
        issues.append(f"{report['duplicate_rows']} duplicate rows detected.")

    # --- Target validity & class balance ---
    unknown = set(df[config.TARGET_COLUMN].unique()) - set(config.TARGET_CLASSES)
    if unknown:
        issues.append(f"Unexpected target labels: {unknown}")
    class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    report["class_balance"] = {k: int(v) for k, v in class_counts.items()}
    minority = min(class_counts.values()) / max(class_counts.values())
    report["minority_majority_ratio"] = round(minority, 3)
    if minority < 0.3:
        issues.append(f"Class imbalance detected (ratio={minority:.2f}).")

    # --- Outliers on continuous columns (IQR rule) ---
    outliers = {}
    for col in config.CONTINUOUS_COLUMNS:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            count = int(mask.sum())
            if count:
                outliers[col] = count
    report["outlier_counts"] = outliers

    report["issues"] = issues
    report["status"] = "PASS_WITH_WARNINGS" if issues else "PASS"

    with open(config.REPORTS_DIR / "data_quality_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    for issue in issues:
        _log(logger, "warning", f"[Quality] {issue}")
    _log(logger, "info", f"Data validation {report['status']}: "
                         f"{report['n_rows']} rows, {report['n_features']} features, "
                         f"{report['total_missing']} missing, "
                         f"{report['duplicate_rows']} duplicates.")
    return report


# ---------------------------------------------------------------------------
# 1.3 Data Pre-processing
# ---------------------------------------------------------------------------
def preprocess_data(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Summary statistics, de-duplication, typed imputation, normalization.

    Continuous columns are median-imputed and Min-Max scaled; categorical codes
    are mode-imputed and left intact. The fitted scaler is persisted so the same
    transformation can be reused at inference time.
    """
    df = df.copy()
    n_before = len(df)

    # --- Summary statistics ---
    summary = df.describe(include="all").transpose()
    summary.to_csv(config.REPORTS_DIR / "summary_statistics.csv")
    _log(logger, "info", "Saved summary statistics for all columns.")

    # --- Data types ---
    dtype_counts = df.dtypes.value_counts().to_dict()
    _log(logger, "info", f"Column data types: "
                         f"{ {str(k): int(v) for k, v in dtype_counts.items()} }")

    # --- De-duplication ---
    df = df.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        _log(logger, "info", f"Removed {removed} duplicate rows.")

    # --- Typed missing-value imputation ---
    continuous = [c for c in config.CONTINUOUS_COLUMNS if c in df.columns]
    categorical = config.categorical_columns(df.columns)
    for col in continuous:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            _log(logger, "info", f"Imputed numeric '{col}' with median={median}.")
    for col in categorical:
        if df[col].isnull().any():
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
            _log(logger, "info", f"Imputed categorical '{col}' with mode={mode}.")

    # --- Normalization of continuous columns only ---
    scaler = MinMaxScaler()
    df[continuous] = scaler.fit_transform(df[continuous])
    joblib.dump(scaler, config.ARTIFACTS_DIR / "minmax_scaler.joblib")
    _log(logger, "info", f"Normalized {len(continuous)} continuous columns to "
                         f"[0, 1] and persisted the scaler.")

    df.to_csv(config.ARTIFACTS_DIR / "preprocessed.csv", index=False)

    # --- Run metadata (lineage for monitoring) ---
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows_in": n_before,
        "rows_out": len(df),
        "duplicates_removed": removed,
        "continuous_columns": len(continuous),
        "categorical_columns": len(categorical),
    }
    with open(config.REPORTS_DIR / "preprocess_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    return df


# ---------------------------------------------------------------------------
# 1.4 Exploratory Data Analysis
# ---------------------------------------------------------------------------
def run_eda(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> dict:
    """Correlation, binning, encoding, feature importance and visualizations."""
    df = df.copy()
    report: dict = {}

    # Raw (unscaled) copy so univariate/bivariate charts show real-world units
    # instead of the normalized [0, 1] values used for modeling.
    raw_df = pd.read_csv(config.RAW_DATA_PATH, sep=config.CSV_SEPARATOR)
    raw_df.columns = [c.strip() for c in raw_df.columns]
    raw_df = raw_df.drop_duplicates().reset_index(drop=True)

    # --- Target encoding: binary Dropout(1) vs Not-Dropout(0) ---
    df["Target_encoded"] = (df[config.TARGET_COLUMN]
                            == config.POSITIVE_LABEL).astype(int)
    report["target_encoding"] = {config.BINARY_CLASSES[0]: 0,
                                 config.BINARY_CLASSES[1]: 1}
    _log(logger, "info", f"Encoded target as binary: "
                         f"{config.POSITIVE_LABEL}=1 vs rest=0.")

    # Numeric feature columns (exclude the target and any derived columns).
    feature_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in (config.TARGET_COLUMN, "Target_encoded")
    ]

    # --- Feature-to-feature correlation matrix ---
    feature_corr = df[feature_cols].corr()
    feature_corr.to_csv(config.REPORTS_DIR / "correlation_matrix.csv")

    # --- Correlation with the binary target (point-biserial) ---
    # With a binary Dropout target we can directly compute the point-biserial
    # (Pearson) correlation of each feature with the Dropout indicator. A
    # positive value means the feature increases with dropout risk.
    target_corr = df[feature_cols].corrwith(df["Target_encoded"])
    target_corr.name = "corr_with_dropout"
    target_corr.sort_values(key=lambda s: s.abs(), ascending=False).to_csv(
        config.REPORTS_DIR / "target_correlation.csv")

    # Rank features by their absolute correlation with dropout.
    strongest = target_corr.abs().sort_values(ascending=False)
    report["top_correlated_features"] = strongest.head(10).round(3).to_dict()
    _log(logger, "info", f"Top features correlated with Dropout: "
                         f"{list(strongest.index[:5])}")

    # --- Binning: bucket Age at enrollment into real-age quartiles ---
    if "Age at enrollment" in raw_df.columns:
        age_bins = pd.qcut(raw_df["Age at enrollment"], q=4,
                           duplicates="drop")
        report["age_bins"] = {str(k): int(v)
                              for k, v in age_bins.value_counts().items()}
        _log(logger, "info", "Created quartile bins for 'Age at enrollment' "
                             "(actual years).")

    # --- Feature importance via a quick RandomForest (binary target) ---
    rf = RandomForestClassifier(
        n_estimators=120, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    rf.fit(df[feature_cols], df["Target_encoded"])
    importances = (
        pd.Series(rf.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    importances.to_csv(config.REPORTS_DIR / "feature_importance.csv")
    report["top_feature_importance"] = importances.head(10).round(4).to_dict()
    _log(logger, "info", f"Most important feature: {importances.index[0]}")

    # --- Visualizations ---
    top_corr_features = strongest.head(12).index
    _make_visualizations(raw_df, target_corr, importances, top_corr_features,
                         logger)

    with open(config.REPORTS_DIR / "eda_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def _make_visualizations(raw_df, target_corr, importances, top_corr_features,
                         logger) -> None:
    """All charts use raw (unscaled) values for interpretability."""
    sns.set_theme(style="whitegrid")

    raw_df = raw_df.copy()
    raw_df["Target (binary)"] = np.where(
        raw_df[config.TARGET_COLUMN] == config.POSITIVE_LABEL,
        config.BINARY_CLASSES[1], config.BINARY_CLASSES[0])
    order = config.BINARY_CLASSES

    # Univariate: binary target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x="Target (binary)", data=raw_df, order=order,
                  hue="Target (binary)", palette="viridis", legend=False)
    plt.title("Univariate analysis: Dropout vs Not-Dropout")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "target_distribution.png", dpi=110)
    plt.close()

    # Univariate: age distribution in actual years
    if "Age at enrollment" in raw_df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(raw_df["Age at enrollment"], bins=30, kde=True, color="teal")
        plt.title("Univariate analysis: Age at enrollment (years)")
        plt.xlabel("Age at enrollment (years)")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "age_distribution.png", dpi=110)
        plt.close()

    # Bivariate: top feature vs binary target, in actual units
    top_feature = importances.index[0]
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Target (binary)", y=top_feature, data=raw_df, order=order,
                hue="Target (binary)", palette="mako", legend=False)
    plt.title(f"Bivariate analysis: {top_feature} vs Dropout")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "bivariate_top_feature.png", dpi=110)
    plt.close()

    # Point-biserial correlation of top features with the Dropout indicator
    corr_top = target_corr.loc[top_corr_features].sort_values()
    plt.figure(figsize=(8, 6))
    colors = ["crimson" if v > 0 else "steelblue" for v in corr_top]
    corr_top.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature correlation with Dropout (point-biserial)")
    plt.xlabel("Correlation with Dropout  (positive \u2192 higher dropout risk)")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "correlation_heatmap.png", dpi=110)
    plt.close()

    # Feature importance bar chart
    plt.figure(figsize=(8, 6))
    importances.head(12).sort_values().plot(kind="barh", color="darkorange")
    plt.title("Top 12 feature importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "feature_importance.png", dpi=110)
    plt.close()

    _log(logger, "info", "Saved 5 EDA visualizations to outputs/plots/.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    _df = ingest_data()
    validate_data(_df)
    _df = preprocess_data(_df)
    run_eda(_df)
    print("Data pipeline finished. See outputs/ for artifacts, reports and plots.")
