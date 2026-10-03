import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    PROJECT_ROOT, RANDOM_STATE, add_engineered_features, build_preprocessor,
    drop_identifier_columns, get_feature_groups, load_data, split_data,
)

warnings.filterwarnings("ignore", category=OptimizeWarning)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_model(name, balanced=False):
    weight = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=weight, random_state=RANDOM_STATE)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, class_weight=weight, random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, class_weight=weight,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)


def make_pipeline(name, strategy, groups):
    steps = [("prep", build_preprocessor(*groups))]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", make_model(name, balanced=(strategy == "class_weight"))))
    return ImbPipeline(steps)


def compare_strategies(X_train, y_train, groups):
    plan = {
        "Logistic Regression": ["Baseline", "class_weight", "SMOTE"],
        "Decision Tree": ["Baseline", "class_weight", "SMOTE"],
        "Random Forest": ["Baseline", "class_weight", "SMOTE"],
        "Gradient Boosting": ["Baseline", "SMOTE"],
    }
    rows = []
    for name, strategies in plan.items():
        for strategy in strategies:
            pipe = make_pipeline(name, strategy, groups)
            res = cross_validate(pipe, X_train, y_train, cv=CV,
                                 scoring=["roc_auc", "precision", "recall", "f1"])
            rows.append({
                "Model": name, "Strategy": strategy,
                "Precision": res["test_precision"].mean(),
                "Recall": res["test_recall"].mean(),
                "F1": res["test_f1"].mean(),
                "ROC-AUC": res["test_roc_auc"].mean(),
            })
    return pd.DataFrame(rows).round(4)


def threshold_sweep(X_train, y_train, groups, thresholds):
    rows = []
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        pipe = make_pipeline(name, "Baseline", groups)
        proba = cross_val_predict(pipe, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
        for t in thresholds:
            pred = (proba >= t).astype(int)
            rows.append({
                "Model": name, "Threshold": t,
                "Precision": precision_score(y_train, pred, zero_division=0),
                "Recall": recall_score(y_train, pred),
                "F1": f1_score(y_train, pred),
                "Flagged_%": pred.mean() * 100,
            })
    return pd.DataFrame(rows).round(4)


def run_imbalance_experiments():
    df = add_engineered_features(drop_identifier_columns(load_data()))
    X_train, _, y_train, _ = split_data(df)
    groups = get_feature_groups(X_train)

    print("Churners in training data: %d of %d (%.1f%%)\n" % (
        y_train.sum(), len(y_train), y_train.mean() * 100))

    strategies = compare_strategies(X_train, y_train, groups)
    print("=== 5-fold CV on training data (threshold 0.5) ===")
    print(strategies.to_string(index=False))

    thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    sweep = threshold_sweep(X_train, y_train, groups, thresholds)
    print("\n=== Threshold sweep (baseline models, out-of-fold predictions) ===")
    print(sweep.to_string(index=False))

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    strategies.to_csv(out_dir / "imbalance_cv_results.csv", index=False)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)


if __name__ == "__main__":
    run_imbalance_experiments()