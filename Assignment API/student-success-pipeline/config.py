"""Central configuration for the Student Success cloud-native pipeline.

All paths are resolved relative to this file so the project runs the same way
locally and inside a cloud worker.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# The raw dataset that was downloaded from the UCI repository lives one level up
# inside the Assignment folder.
RAW_DATA_PATH = PROJECT_ROOT.parent / "data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

for _d in (OUTPUT_DIR, PLOTS_DIR, ARTIFACTS_DIR, LOGS_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset details
# ---------------------------------------------------------------------------
# The UCI file uses ';' as the separator.
CSV_SEPARATOR = ";"
TARGET_COLUMN = "Target"
# Original class labels in the dataset.
TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]
# The raw dataset is 3-class, but the task is framed as a binary problem:
# Dropout (the event we want to flag early) vs everyone who did not drop out.
POSITIVE_LABEL = "Dropout"
BINARY_CLASSES = ["Not-Dropout", "Dropout"]  # encoded as 0 and 1 respectively

# Continuous numeric columns (grades, counts, macro-economic indicators). Every
# other feature in this dataset is a categorical code, so the two groups are
# pre-processed differently (scale continuous, leave categorical codes intact).
CONTINUOUS_COLUMNS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Minimum number of rows for a "meaningful" experiment (data-quality gate).
MIN_EXPECTED_ROWS = 1000
EXPECTED_FEATURE_COUNT = 36  # 37 columns minus the target

# ---------------------------------------------------------------------------
# Categorical feature encoding (ML pipeline)
# ---------------------------------------------------------------------------
# Low-cardinality categoricals are dummy/one-hot encoded; high-cardinality
# nominals are Weight-of-Evidence encoded against the Dropout event.
DUMMY_COLUMNS = [
    "Marital status",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
WOE_COLUMNS = [
    "Application mode",
    "Application order",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]
# WoE is a binary-event measure; the event of interest is "Dropout", whose
# encoded label is 1 in the binary target (see BINARY_CLASSES ordering).
WOE_EVENT_CLASS_INDEX = 1

# ---------------------------------------------------------------------------
# Machine learning configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70 / 30 train-test split as required by the assignment
CV_FOLDS = 5  # Stratified k-fold cross-validation
TUNING_ITERATIONS = 6  # RandomizedSearchCV samples per model
# XGBoost gets a deeper, dedicated randomized hyper-parameter search.
XGB_TUNING_ITERATIONS = 30

# ---------------------------------------------------------------------------
# MLflow experiment tracking (MLOps)
# ---------------------------------------------------------------------------
# SQLite tracking backend (the file store is deprecated / maintenance-mode in
# MLflow 3.x). Runs, params and metrics live in outputs/mlflow.db and artifacts
# in outputs/mlartifacts, so everything is local and offline — no server needed.
MLFLOW_TRACKING_URI = f"sqlite:///{OUTPUT_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_URI = (OUTPUT_DIR / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT = "student-success-dropout"


def categorical_columns(all_columns) -> list:
    """Feature columns that are not continuous and not the target."""
    return [
        c
        for c in all_columns
        if c not in CONTINUOUS_COLUMNS and c != TARGET_COLUMN
    ]

# ---------------------------------------------------------------------------
# DataOps / scheduling
# ---------------------------------------------------------------------------
# The assignment asks the data pipeline to run every 2 minutes.
SCHEDULE_INTERVAL_SECONDS = 120
