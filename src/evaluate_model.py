import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.preprocessing import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, threshold=0.5):
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred),
        "F1": f1_score(y_test, test_pred),
        "ROC-AUC": roc_auc_score(y_test, test_proba),
        "Train ROC-AUC": roc_auc_score(y_train, train_proba),
    }


def results_table(results):
    table = pd.DataFrame(results).T.round(4)
    table.index.name = "Model"
    return table


def plot_confusion_matrices(fitted, X_test, y_test, filename, threshold=0.5):
    n = len(fitted)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = axes.ravel()
    for ax, (name, pipe) in zip(axes, fitted.items()):
        pred = (pipe.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Stay", "Churn"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def plot_roc_curves(fitted, X_test, y_test, filename):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set_title("ROC Curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)