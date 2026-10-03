import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score,
)
from sklearn.pipeline import Pipeline

from src.evaluate_model import FIG_DIR, evaluate_model, results_table
from src.imbalance_experiments import make_model
from src.preprocessing import (
    PROJECT_ROOT, RANDOM_STATE, add_engineered_features, build_preprocessor,
    drop_identifier_columns, get_feature_groups, load_data, split_data,
)

warnings.filterwarnings("ignore", category=OptimizeWarning)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]


def build_pipeline(model, groups):
    return Pipeline([("prep", build_preprocessor(*groups)), ("model", model)])


def tune_gradient_boosting(X_train, y_train, groups):
    search = RandomizedSearchCV(
        build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE), groups),
        param_distributions={
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [10, 20, 50],
        },
        n_iter=10, scoring="roc_auc", cv=CV,
        random_state=RANDOM_STATE, n_jobs=-1, refit=False,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


def choose_threshold(pipeline, X_train, y_train):
    proba = cross_val_predict(pipeline, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.10, 0.61, 0.01), 2)
    scores = [f1_score(y_train, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))]), float(max(scores))


def bootstrap_ci(pipeline, X_test, y_test, threshold, n_boot=1000):
    proba = pipeline.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    stats = {"Precision": [], "Recall": [], "F1": [], "ROC-AUC": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        pred = (proba[idx] >= threshold).astype(int)
        stats["Precision"].append(precision_score(y[idx], pred, zero_division=0))
        stats["Recall"].append(recall_score(y[idx], pred))
        stats["F1"].append(f1_score(y[idx], pred))
        stats["ROC-AUC"].append(roc_auc_score(y[idx], proba[idx]))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in stats.items()}


def plot_confusion_grid(fitted, thresholds, X_test, y_test, filename):
    fig, axes = plt.subplots(1, len(fitted), figsize=(4.2 * len(fitted), 4))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        pred = (pipe.predict_proba(X_test)[:, 1] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Stay", "Churn"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"{name}\n(threshold {thresholds[name]:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def plot_roc(fitted, X_test, y_test, filename):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set_title("ROC Curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def run_final_evaluation():
    df = add_engineered_features(drop_identifier_columns(load_data()))
    X_train, X_test, y_train, y_test = split_data(df)
    groups = get_feature_groups(X_train)

    print("--- Tuning Gradient Boosting (training data only) ---")
    untuned_auc = cross_val_score(
        build_pipeline(make_model("Gradient Boosting"), groups),
        X_train, y_train, cv=CV, scoring="roc_auc").mean()
    best_params, tuned_auc = tune_gradient_boosting(X_train, y_train, groups)
    print("Untuned CV ROC-AUC: %.4f | Tuned CV ROC-AUC: %.4f" % (untuned_auc, tuned_auc))
    print("Best parameters:", best_params)

    models = {name: make_model(name) for name in MODEL_NAMES}
    if tuned_auc > untuned_auc:
        tuned = GradientBoostingClassifier(random_state=RANDOM_STATE)
        tuned.set_params(**{k.replace("model__", ""): v for k, v in best_params.items()})
        models["Gradient Boosting"] = tuned
        print("-> Using the tuned Gradient Boosting.")
    else:
        print("-> Tuning did not improve CV ROC-AUC, keeping default Gradient Boosting.")

    print("\n--- Choosing thresholds on training data (out-of-fold, max F1) ---")
    thresholds = {}
    for name, model in models.items():
        t, f1 = choose_threshold(build_pipeline(model, groups), X_train, y_train)
        thresholds[name] = t
        print(f"{name:20s} threshold = {t:.2f}  (training CV F1 = {f1:.4f})")

    fitted, results = {}, {}
    for name, model in models.items():
        pipe = build_pipeline(model, groups).fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {
            **evaluate_model(pipe, X_train, y_train, X_test, y_test, thresholds[name]),
            "Threshold": thresholds[name],
        }
    table = results_table(results).drop(columns=["Train ROC-AUC"])
    print("\n=== TEST SET, each model at its pre-chosen threshold ===")
    print(table.to_string())

    final_name = "Gradient Boosting"
    final = fitted[final_name]
    t = thresholds[final_name]
    proba = final.predict_proba(X_test)[:, 1]
    pred = (proba >= t).astype(int)
    print(f"\n=== FINAL MODEL: {final_name} (threshold {t:.2f}) ===")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"], digits=3))
    tn = int(((pred == 0) & (y_test == 0)).sum()); fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum()); tp = int(((pred == 1) & (y_test == 1)).sum())
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("Customers flagged: %.1f%%" % (pred.mean() * 100))

    at_half = evaluate_model(final, X_train, y_train, X_test, y_test, 0.5)
    print("Same model at threshold 0.50 -> Precision %.3f | Recall %.3f | F1 %.3f" % (
        at_half["Precision"], at_half["Recall"], at_half["F1"]))

    print("\n95% bootstrap intervals on the test set (final model):")
    for k, (lo, hi) in bootstrap_ci(final, X_test, y_test, t).items():
        print(f"  {k:10s} {lo:.3f} - {hi:.3f}")

    plot_confusion_grid(fitted, thresholds, X_test, y_test, "22_final_confusion_matrices.png")
    plot_roc(fitted, X_test, y_test, "23_final_roc_curves.png")
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "final_results.csv")


if __name__ == "__main__":
    run_final_evaluation()