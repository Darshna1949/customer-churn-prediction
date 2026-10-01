import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import (
    PROJECT_ROOT, TARGET, drop_identifier_columns, load_data, split_data,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C78A8", 1: "#E45756"}


def get_training_frame():
    df = drop_identifier_columns(load_data())
    X_train, _, y_train, _ = split_data(df)
    return pd.concat([X_train, y_train], axis=1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


def churn_rate_table(df, col):
    table = df.groupby(col, observed=True)[TARGET].agg(customers="count", churn_rate="mean")
    table["churn_rate"] = (table["churn_rate"] * 100).round(1)
    return table


def plot_churn_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Stayed (0)", "Churned (1)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df) * 100:.1f}%)", ha="center", va="bottom")
    ax.set_title("Churn Distribution (training data)")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "01_churn_distribution.png")


def plot_churn_rate_bar(df, col, filename, title, order=None):
    table = churn_rate_table(df, col)
    if order is not None:
        table = table.loc[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=table.index.astype(str), y=table["churn_rate"], color="#E45756", ax=ax)
    ax.axhline(df[TARGET].mean() * 100, color="black", linestyle="--", label="Overall churn rate")
    for i, v in enumerate(table["churn_rate"]):
        ax.text(i, v + 0.5, f"{v}%", ha="center")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Churn rate (%)")
    ax.legend()
    save(fig, filename)
    return table


def plot_numeric_distribution(df, col, filename, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=col, hue=TARGET, bins=30, kde=True, palette=PALETTE,
                 stat="density", common_norm=False, ax=ax)
    ax.set_title(title)
    save(fig, filename)


def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numerical features)")
    save(fig, "12_correlation_heatmap.png")
    return numeric.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)


def plot_combination_heatmap(df, row, col, filename, title):
    pivot = df.pivot_table(index=row, columns=col, values=TARGET, aggfunc="mean", observed=True) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={"label": "Churn rate (%)"})
    ax.set_title(title)
    save(fig, filename)
    return pivot.round(1)


def run_eda():
    df = get_training_frame()
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 40, 50, 60, 100],
                             labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    df["balance_status"] = (df["balance"] > 0).map({False: "Zero balance", True: "Has balance"})
    df["balance_band"] = pd.cut(df["balance"], bins=[-1, 0, 100000, 150000, 300000],
                                labels=["Zero", "1-100k", "100k-150k", "150k+"])
    df["salary_band"] = pd.qcut(df["estimated_salary"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    df["score_band"] = pd.qcut(df["credit_score"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])

    print("Overall churn rate (train): %.1f%%\n" % (df[TARGET].mean() * 100))
    plot_churn_distribution(df)

    tables = {}
    plan = [
        ("age_group", "02_churn_by_age_group.png", "Churn Rate by Age Group", None),
        ("gender", "04_churn_by_gender.png", "Churn Rate by Gender", None),
        ("country", "05_churn_by_country.png", "Churn Rate by Country", None),
        ("tenure", "06_churn_by_tenure.png", "Churn Rate by Tenure (years)", None),
        ("balance_band", "07_churn_by_balance.png", "Churn Rate by Balance Band", None),
        ("products_number", "08_churn_by_products.png", "Churn Rate by Number of Products", None),
        ("active_member", "09_churn_by_active_member.png", "Churn Rate by Active Membership", None),
        ("credit_card", "10_churn_by_credit_card.png", "Churn Rate by Credit Card Ownership", None),
        ("salary_band", "11_churn_by_salary.png", "Churn Rate by Salary Quintile", None),
        ("score_band", "13_churn_by_credit_score.png", "Churn Rate by Credit Score Quintile", None),
    ]
    for col, fname, title, order in plan:
        tables[col] = plot_churn_rate_bar(df, col, fname, title, order)

    plot_numeric_distribution(df, "age", "03_age_distribution.png", "Age Distribution: Stayed vs Churned")
    corr = plot_correlation_heatmap(df)

    combos = {
        "country x active_member": plot_combination_heatmap(df, "country", "active_member", "14_country_x_active.png", "Churn % : Country x Active Member"),
        "age_group x products_number": plot_combination_heatmap(df, "age_group", "products_number", "15_age_x_products.png", "Churn % : Age Group x Products"),
        "age_group x active_member": plot_combination_heatmap(df, "age_group", "active_member", "16_age_x_active.png", "Churn % : Age Group x Active Member"),
        "country x balance_status": plot_combination_heatmap(df, "country", "balance_status", "17_country_x_balance.png", "Churn % : Country x Zero/Has Balance"),
    }

    for name, t in tables.items():
        print(f"--- {name} ---")
        print(t.to_string(), "\n")
    print("--- balance_status ---")
    print(churn_rate_table(df, "balance_status").to_string(), "\n")
    print("--- correlation with churn ---")
    print(corr.round(3).to_string(), "\n")
    for name, t in combos.items():
        print(f"--- {name} ---")
        print(t.to_string(), "\n")
    print("Mean age  stayed vs churned:", df.groupby(TARGET)["age"].mean().round(1).to_dict())
    print("Mean balance stayed vs churned:", df.groupby(TARGET)["balance"].mean().round(0).to_dict())


if __name__ == "__main__":
    run_eda()