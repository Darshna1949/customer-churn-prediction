from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Bank_Customer_Churn_Prediction.csv"

TARGET = "churn"
ID_COLUMNS = ["customer_id"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def drop_identifier_columns(df):
    return df.drop(columns=ID_COLUMNS)


def add_engineered_features(df):
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"], bins=[-np.inf, 30, 40, 50, 60, np.inf],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    ).astype(object)
    df["has_zero_balance"] = (df["balance"] == 0).astype(int)
    df["products_group"] = pd.cut(
        df["products_number"], bins=[-np.inf, 1, 2, np.inf],
        labels=["1 product", "2 products", "3-4 products"],
    ).astype(object)
    return df


def get_feature_groups(X):
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    binary = [c for c in numeric if X[c].nunique() == 2]
    numerical = [c for c in numeric if c not in binary]
    return numerical, categorical, binary


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(numerical, categorical, binary):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 drop="if_binary", sparse_output=False)),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical),
        ("bin", binary_pipe, binary),
    ])


if __name__ == "__main__":
    df = drop_identifier_columns(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    numerical, categorical, binary = get_feature_groups(X_train)

    print("Numerical  :", numerical)
    print("Categorical:", categorical)
    print("Binary     :", binary)

    print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)
    print("Churn % in train:", round(y_train.mean() * 100, 2))
    print("Churn % in test :", round(y_test.mean() * 100, 2))

    preprocessor = build_preprocessor(numerical, categorical, binary)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    names = preprocessor.get_feature_names_out()
    print("\nFeatures after preprocessing:", X_train_t.shape[1])
    print(list(names))

    train_out = pd.DataFrame(X_train_t, columns=names)
    print("\nTrain mean of scaled numeric columns (should be ~0):")
    print(train_out.filter(like="num__").mean().round(3))
    print("\nTest mean of scaled numeric columns (close to, not exactly, 0):")
    print(pd.DataFrame(X_test_t, columns=names).filter(like="num__").mean().round(3))