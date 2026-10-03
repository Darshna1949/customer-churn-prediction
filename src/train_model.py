import warnings

import joblib
import pandas as pd
import sklearn
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import (
    DECISION_THRESHOLD, FEATURE_INFO_PATH, FINAL_MODEL_PARAMS, MODEL_DIR,
    MODEL_PATH, REPORT_DIR, RISK_HIGH_MIN, RISK_LOW_MAX,
)
from src.evaluate_model import evaluate_model
from src.preprocessing import (
    ID_COLUMNS, RANDOM_STATE, TARGET, add_engineered_features, build_preprocessor,
    drop_identifier_columns, get_feature_groups, load_data, split_data,
)

warnings.filterwarnings("ignore", category=OptimizeWarning)


def aggregate_impurity_importance(pipeline, model_columns):
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    totals = dict.fromkeys(model_columns, 0.0)
    for name, value in zip(names, importances):
        clean = name.split("__", 1)[1]
        match = max((c for c in model_columns if clean == c or clean.startswith(c + "_")), key=len)
        totals[match] += float(value)
    return totals


def build_feature_info(raw_df, X_train, groups, pipeline, metrics):
    numerical, categorical, binary = groups
    input_columns = [c for c in raw_df.columns if c not in ID_COLUMNS + [TARGET]]
    numeric_inputs = [c for c in input_columns if c in numerical]

    reference = {}
    for col in input_columns:
        if col in numerical:
            reference[col] = float(X_train[col].median())
        else:
            value = X_train[col].mode().iloc[0]
            reference[col] = value.item() if hasattr(value, "item") else value

    return {
        "input_columns": input_columns,
        "model_columns": list(X_train.columns),
        "numerical": numerical,
        "categorical": categorical,
        "binary": binary,
        "categories": {c: sorted(raw_df[c].unique().tolist())
                       for c in input_columns if c in categorical},
        "ranges": {c: {"min": float(raw_df[c].min()), "max": float(raw_df[c].max())}
                   for c in numeric_inputs},
        "reference_values": reference,
        "decision_threshold": DECISION_THRESHOLD,
        "risk_cutoffs": {"low_max": RISK_LOW_MAX, "high_min": RISK_HIGH_MIN},
        "risk_note": "Risk thresholds are project-defined, not universal business standards.",
        "metrics": metrics,
        "dataset": {"rows": int(len(raw_df)), "churn_rate": float(raw_df[TARGET].mean())},
        "transformed_feature_names": pipeline.named_steps["prep"].get_feature_names_out().tolist(),
        "model_params": FINAL_MODEL_PARAMS,
        "sklearn_version": sklearn.__version__,
    }


def train_and_save():
    raw_df = drop_identifier_columns(load_data())
    df = add_engineered_features(raw_df)
    X_train, X_test, y_train, y_test = split_data(df)
    groups = get_feature_groups(X_train)

    pipeline = Pipeline([
        ("prep", build_preprocessor(*groups)),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE, **FINAL_MODEL_PARAMS)),
    ]).fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test, DECISION_THRESHOLD)
    metrics.pop("Train ROC-AUC")
    metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                    "Flagged_%": float(pred.mean() * 100)})
    metrics = {k: float(v) for k, v in metrics.items()}

    print("Final model: Gradient Boosting | decision threshold = %.2f" % DECISION_THRESHOLD)
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    for t in (0.28, 0.30, 0.50):
        m = evaluate_model(pipeline, X_train, y_train, X_test, y_test, t)
        print("  threshold %.2f -> precision %.3f | recall %.3f | F1 %.3f | accuracy %.3f" % (
            t, m["Precision"], m["Recall"], m["F1"], m["Accuracy"]))

    impurity = aggregate_impurity_importance(pipeline, list(X_train.columns))
    perm = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "impurity_importance": [impurity[c] for c in X_test.columns],
        "permutation_auc_drop": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("permutation_auc_drop", ascending=False).round(4)
    print("\nFeature importance (impurity: from training | permutation: ROC-AUC drop on test set)")
    print(importance.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(build_feature_info(raw_df, X_train, groups, pipeline, metrics), FEATURE_INFO_PATH)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    pd.DataFrame({"churn": y_test.to_numpy(), "churn_probability": proba}).to_csv(
        REPORT_DIR / "test_predictions.csv", index=False)
    print("\nSaved:", MODEL_PATH.name, "|", FEATURE_INFO_PATH.name,
          "| feature_importance.csv | test_predictions.csv")


if __name__ == "__main__":
    train_and_save()