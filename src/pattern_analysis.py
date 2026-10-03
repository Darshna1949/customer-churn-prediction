import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier, export_text

from src.eda import get_training_frame
from src.preprocessing import RANDOM_STATE, TARGET


def add_exploration_groups(df):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 40, 50, 60, 100],
                             labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    df["balance_status"] = np.where(df["balance"] > 0, "Has balance", "Zero balance")
    df["salary_band"] = pd.qcut(df["estimated_salary"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    df["score_band"] = pd.qcut(df["credit_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def association_tests(df, columns):
    rows = []
    for col in columns:
        table = pd.crosstab(df[col], df[TARGET])
        chi2, p_value, _, _ = chi2_contingency(table)
        cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        rows.append({"feature": col, "p_value": p_value, "cramers_v": round(cramers_v, 3)})
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    result["significant_at_5%"] = result["p_value"] < 0.05
    return result.reset_index(drop=True)


def segment_report(df, segments):
    baseline = df[TARGET].mean()
    rows = []
    for name, mask in segments.items():
        n = int(mask.sum())
        rate = df.loc[mask, TARGET].mean()
        margin = 1.96 * np.sqrt(rate * (1 - rate) / n)
        rows.append({
            "segment": name,
            "customers": n,
            "share_of_all_%": round(n / len(df) * 100, 1),
            "churn_rate_%": round(rate * 100, 1),
            "ci_95_%": f"{max(rate - margin, 0) * 100:.1f}-{min(rate + margin, 1) * 100:.1f}",
            "lift": round(rate / baseline, 2),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def extract_tree_rules(df, max_depth=3, min_samples_leaf=100):
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=["country", "gender"], dtype=int)
    y = df[TARGET]
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  random_state=RANDOM_STATE).fit(X, y)
    rules = export_text(tree, feature_names=list(X.columns), show_weights=True)
    leaves = pd.DataFrame({"leaf": tree.apply(X), "churn": y.values})
    leaf_summary = leaves.groupby("leaf")["churn"].agg(customers="count", churn_rate="mean")
    leaf_summary["churn_rate_%"] = (leaf_summary.pop("churn_rate") * 100).round(1)
    return rules, leaf_summary.sort_values("churn_rate_%", ascending=False)


def run_pattern_analysis():
    raw = get_training_frame()
    df = add_exploration_groups(raw)
    baseline = df[TARGET].mean()
    print(f"Baseline churn rate (training data): {baseline * 100:.1f}%\n")

    print("=== 1. Is the association real or just chance? (chi-square + Cramer's V) ===")
    cols = ["age_group", "products_number", "country", "active_member", "gender",
            "balance_status", "score_band", "salary_band", "tenure", "credit_card"]
    print(association_tests(df, cols).to_string(), "\n")

    print("=== 2. Segment report (lift = segment churn rate / baseline churn rate) ===")
    active = df["active_member"] == 1
    inactive = ~active
    segments = {
        "Age 41-60": df["age"].between(41, 60),
        "Age 41-60 AND inactive": df["age"].between(41, 60) & inactive,
        "Age 41-60 AND Germany": df["age"].between(41, 60) & (df["country"] == "Germany"),
        "Germany AND inactive": (df["country"] == "Germany") & inactive,
        "Germany AND female": (df["country"] == "Germany") & (df["gender"] == "Female"),
        "Inactive": inactive,
        "1 product": df["products_number"] == 1,
        "1 product AND age 41-60": (df["products_number"] == 1) & df["age"].between(41, 60),
        "3-4 products": df["products_number"] >= 3,
        "Germany": df["country"] == "Germany",
        "Female": df["gender"] == "Female",
        "2 products": df["products_number"] == 2,
        "Age <= 40 AND active AND 2 products": (df["age"] <= 40) & active & (df["products_number"] == 2),
        "Age <= 40 AND active": (df["age"] <= 40) & active,
    }
    print(segment_report(df, segments).to_string(), "\n")

    print("=== 3. Rules extracted by a shallow decision tree (training data only) ===")
    rules, leaf_summary = extract_tree_rules(raw)
    print(rules)
    print(leaf_summary.to_string())


if __name__ == "__main__":
    run_pattern_analysis()