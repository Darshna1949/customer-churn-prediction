import warnings

import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_model import (
    evaluate_model, plot_confusion_matrices, plot_roc_curves, results_table,
)
from src.preprocessing import (
    PROJECT_ROOT, RANDOM_STATE, add_engineered_features, build_preprocessor,
    drop_identifier_columns, get_feature_groups, load_data, split_data,
)

warnings.filterwarnings("ignore", category=OptimizeWarning)


def get_models():
    return {
        "Dummy (always no churn)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Decision Tree (unlimited depth)": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def run_baseline():
    df = drop_identifier_columns(load_data())
    df = add_engineered_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    numerical, categorical, binary = get_feature_groups(X_train)

    fitted, results = {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([
            ("prep", build_preprocessor(numerical, categorical, binary)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = evaluate_model(pipe, X_train, y_train, X_test, y_test)

    table = results_table(results)
    print("Test set size:", len(y_test), "| churners:", int(y_test.sum()))
    print(table.to_string())

    shown = {k: v for k, v in fitted.items() if k != "Decision Tree (unlimited depth)"}
    plot_confusion_matrices(shown, X_test, y_test, "20_baseline_confusion_matrices.png")
    plot_roc_curves(shown, X_test, y_test, "21_baseline_roc_curves.png")

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "baseline_results.csv")


if __name__ == "__main__":
    run_baseline()