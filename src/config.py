from src.preprocessing import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.pkl"

DECISION_THRESHOLD = 0.30
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.60

FINAL_MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 1.0,
    "min_samples_leaf": 20,
}