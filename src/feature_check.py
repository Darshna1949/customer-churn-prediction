import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    RANDOM_STATE, add_engineered_features, build_preprocessor,
    drop_identifier_columns, get_feature_groups, load_data, split_data,
)


def cv_scores(X_train, y_train, model):
    numerical, categorical, binary = get_feature_groups(X_train)
    pipe = Pipeline([
        ("prep", build_preprocessor(numerical, categorical, binary)),
        ("model", model),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring=["roc_auc", "recall", "f1"])
    return {k: round(res[f"test_{k}"].mean(), 4) for k in ["roc_auc", "recall", "f1"]}


def run_feature_check():
    df = drop_identifier_columns(load_data())
    X_train, _, y_train, _ = split_data(df)
    X_train_fe = add_engineered_features(X_train)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in models.items():
        for label, X in [("original", X_train), ("engineered", X_train_fe)]:
            rows.append({"model": name, "features": label, **cv_scores(X_train if label == "original" else X_train_fe, y_train, model)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run_feature_check()