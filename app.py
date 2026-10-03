import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.config import MODEL_PATH, REPORT_DIR
from src.eda import get_training_frame
from src.pattern_analysis import add_exploration_groups, association_tests
from src.prediction import load_artifacts, predict_customer
from src.preprocessing import PROJECT_ROOT, TARGET

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

RED = "#E45756"
BLUE = "#4C78A8"

FEATURE_VIEWS = {
    "Age group": ("age_group", "Churn rises sharply in the middle age groups and is lowest among younger customers. Age is the strongest single pattern in this dataset."),
    "Number of products": ("products_number", "A U-shaped pattern: customers with 2 products churn least, while 1 product is higher and 3-4 products are far higher. The 3-4 product groups are small, so treat their rates with caution."),
    "Active membership": ("active_member", "Inactive customers churn more than active ones. Inactivity may be a warning sign rather than a reason for leaving."),
    "Country": ("country", "Germany shows a noticeably higher churn rate than France and Spain. The dataset has no information on why."),
    "Gender": ("gender", "Female customers churn somewhat more than male customers in this data."),
    "Balance (zero vs has balance)": ("balance_status", "Customers with a zero balance churn less than those holding a balance. Note that no customer in Germany has a zero balance, so country and balance overlap."),
    "Tenure (years)": ("tenure", "No clear pattern: churn stays in a narrow band across tenure values."),
    "Credit card ownership": ("credit_card", "Almost no difference between customers with and without a credit card."),
    "Credit score (quintiles)": ("score_band", "Only a very weak pattern. Credit score is not a strong signal on its own."),
    "Estimated salary (quintiles)": ("salary_band", "No clear pattern across salary groups."),
}

CAUSATION_NOTE = (
    "These charts show **observed patterns** in historical data. They show association, "
    "not causation: a group that churns more is not necessarily leaving *because* of that trait."
)


@st.cache_data
def read_report(name):
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_training_data():
    return add_exploration_groups(get_training_frame())


def require_model():
    if not MODEL_PATH.exists():
        st.error("The trained model was not found. Run `python -m src.train_model` first.")
        st.stop()
    return load_artifacts()


def bar_figure(table, xlabel, baseline):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(table.index.astype(str), table["churn_rate"], color=RED)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label="Overall churn rate")
    for i, v in enumerate(table["churn_rate"]):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Churn rate (%)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def heatmap_figure(pivot, title):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    image = ax.imshow(pivot.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, label="Churn rate (%)")
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def dashboard_page():
    _, info = require_model()
    metrics = info["metrics"]
    churn_rate = info["dataset"]["churn_rate"]
    rows = info["dataset"]["rows"]

    st.title("Customer Churn Prediction Using Behavioral Patterns")
    st.write(
        "This project recognises behavioural patterns in historical bank customer data "
        "(demographics, account details and activity) and uses them to estimate the "
        "probability that a customer will leave the bank."
    )

    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{rows:,}")
    c2.metric("Overall churn rate", f"{churn_rate * 100:.1f}%")
    c3.metric("Customers who churned", f"{round(rows * churn_rate):,}")
    c4.metric("Input features", len(info["input_columns"]))
    st.caption(
        f"Roughly 1 in 5 customers churned. A model that always predicts 'stays' would be right "
        f"{(1 - churn_rate) * 100:.1f}% of the time while finding no churners, which is why accuracy "
        f"alone is not used to judge the models."
    )

    st.subheader("Model performance summary")
    st.write(
        f"**Final model:** Gradient Boosting, evaluated on 2,000 unseen test customers "
        f"at a decision threshold of {info['decision_threshold']:.2f}."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['Precision']:.3f}")
    m3.metric("Recall", f"{metrics['Recall']:.3f}")
    m4.metric("F1-score", f"{metrics['F1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    st.subheader("How risk is reported")
    cutoffs = info["risk_cutoffs"]
    st.table(pd.DataFrame({
        "Churn probability": [
            f"below {cutoffs['low_max'] * 100:.0f}%",
            f"{cutoffs['low_max'] * 100:.0f}% to below {cutoffs['high_min'] * 100:.0f}%",
            f"{cutoffs['high_min'] * 100:.0f}% and above",
        ],
        "Risk level": ["Low", "Medium", "High"],
        "Prediction": ["Likely to Stay", "Likely to Churn", "Likely to Churn"],
    }))
    st.caption(info["risk_note"])


def prediction_page():
    _, info = require_model()
    ref, rng, cats = info["reference_values"], info["ranges"], info["categories"]

    st.title("Churn Prediction")
    st.write("Enter a customer's details. Values are limited to the ranges seen in the training data.")

    with st.form("customer_form"):
        left, right = st.columns(2)
        with left:
            country = st.selectbox("Country", cats["country"], index=cats["country"].index(ref["country"]))
            gender = st.selectbox("Gender", cats["gender"], index=cats["gender"].index(ref["gender"]))
            age = st.number_input("Age", int(rng["age"]["min"]), int(rng["age"]["max"]), int(ref["age"]))
            tenure = st.slider("Tenure (years with the bank)", int(rng["tenure"]["min"]),
                               int(rng["tenure"]["max"]), int(ref["tenure"]))
            credit_score = st.slider("Credit score", int(rng["credit_score"]["min"]),
                                     int(rng["credit_score"]["max"]), int(ref["credit_score"]))
        with right:
            balance = st.number_input("Account balance", float(rng["balance"]["min"]),
                                      float(rng["balance"]["max"]), float(round(ref["balance"])), step=1000.0)
            products = st.selectbox("Number of products", [1, 2, 3, 4], index=int(ref["products_number"]) - 1)
            credit_card = st.selectbox("Has a credit card", ["Yes", "No"], index=0 if ref["credit_card"] == 1 else 1)
            active = st.selectbox("Active member", ["Yes", "No"], index=0 if ref["active_member"] == 1 else 1)
            salary = st.number_input("Estimated salary", float(rng["estimated_salary"]["min"]),
                                     float(rng["estimated_salary"]["max"]),
                                     float(round(ref["estimated_salary"])), step=1000.0)
        submitted = st.form_submit_button("PREDICT CHURN", type="primary")

    if not submitted:
        return

    customer = {
        "credit_score": credit_score, "country": country, "gender": gender, "age": age,
        "tenure": tenure, "balance": balance, "products_number": products,
        "credit_card": 1 if credit_card == "Yes" else 0,
        "active_member": 1 if active == "Yes" else 0, "estimated_salary": salary,
    }
    result = predict_customer(customer)

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customer Churn Probability", f"{result['probability_pct']:.1f}%")
    c2.metric("Prediction", result["prediction"])
    c3.metric("Risk Level", result["risk_level"])
    st.progress(min(max(result["probability"], 0.0), 1.0))

    message = f"{result['risk_level']} risk: churn probability {result['probability_pct']:.1f}%."
    if result["risk_level"] == "Low":
        st.success(message)
    elif result["risk_level"] == "Medium":
        st.warning(message)
    else:
        st.error(message)

    st.subheader("Factors associated with this prediction")
    factors = result["top_factors"]
    if factors:
        table = pd.DataFrame({
            "Feature": [f["feature"].replace("_", " ") for f in factors],
            "This customer": [str(f["value"]) for f in factors],
            "Typical customer": [str(round(f["typical_value"], 1)) if isinstance(f["typical_value"], float)
                                 else str(f["typical_value"]) for f in factors],
            "Effect on probability (points)": [f["effect_pct_points"] for f in factors],
            "Direction": [f["direction"] for f in factors],
        })
        st.dataframe(table, hide_index=True)
    else:
        st.write("No single input moves this prediction by more than 1 percentage point.")
    st.caption(
        "Each effect shows how the probability changes when only that input is replaced by the "
        "typical training value. Effects are computed one at a time, so they do not add up to the "
        "total. They describe what the model has learned, not what causes a customer to leave."
    )
    st.caption(info["risk_note"])


def pattern_page():
    st.title("Pattern Analysis")
    if not (PROJECT_ROOT / "data").exists():
        st.error("The data folder was not found.")
        st.stop()
    df = load_training_data()
    baseline = df[TARGET].mean() * 100

    st.write(
        f"Patterns below come from the **training data only** ({len(df):,} customers), "
        f"so the test set stays unseen. Overall churn rate: **{baseline:.1f}%**."
    )
    st.info(CAUSATION_NOTE)

    st.subheader("Churn distribution")
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["Stayed", "Churned"], counts.values, color=[BLUE, RED])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Customers")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    show(fig)

    st.subheader("Churn rate by behavioural feature")
    label = st.selectbox("Choose a feature", list(FEATURE_VIEWS))
    column, note = FEATURE_VIEWS[label]
    table = df.groupby(column, observed=True)[TARGET].agg(customers="count", churn_rate="mean")
    table["churn_rate"] = table["churn_rate"] * 100
    left, right = st.columns([3, 2])
    with left:
        show(bar_figure(table, label, baseline))
    with right:
        st.dataframe(table.round({"churn_rate": 1}).rename(
            columns={"customers": "Customers", "churn_rate": "Churn rate (%)"}))
        highest = table["churn_rate"].idxmax()
        lowest = table["churn_rate"].idxmin()
        st.write(f"Highest: **{highest}** ({table['churn_rate'].max():.1f}%). "
                 f"Lowest: **{lowest}** ({table['churn_rate'].min():.1f}%).")
    st.write(note)
    if table["customers"].min() < 300:
        st.caption("Some groups are small (under 300 customers), so their rates are less reliable.")

    st.subheader("Combinations of features")
    c1, c2 = st.columns(2)
    with c1:
        pivot = df.pivot_table(index="age_group", columns="active_member", values=TARGET,
                               aggfunc="mean", observed=True) * 100
        show(heatmap_figure(pivot, "Churn % by age group and active membership (0 = inactive)"))
    with c2:
        pivot = df.pivot_table(index="country", columns="active_member", values=TARGET,
                               aggfunc="mean") * 100
        show(heatmap_figure(pivot, "Churn % by country and active membership (0 = inactive)"))
    st.write(
        "Inactive customers churn more in every group, and the gap is largest among older customers. "
        "Patterns that combine features are often stronger than any single feature."
    )

    st.subheader("How strong is each pattern?")
    columns = ["age_group", "products_number", "country", "active_member", "gender",
               "balance_status", "score_band", "salary_band", "tenure", "credit_card"]
    strength = association_tests(df, columns)
    strength["p_value"] = strength["p_value"].map(lambda p: f"{p:.2g}")
    strength = strength.rename(columns={
        "feature": "Feature", "p_value": "p-value", "cramers_v": "Cramér's V (strength)",
        "significant_at_5%": "Statistically significant (5%)"})
    st.dataframe(strength, hide_index=True)
    st.caption(
        "A chi-square test checks whether a pattern could be chance; Cramér's V (0 to 1) measures how "
        "strong it is. Tenure, salary and credit card ownership show no reliable pattern."
    )

    importance = read_report("feature_importance.csv")
    if importance is not None:
        st.subheader("Which features does the final model rely on?")
        shown = importance.head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(shown["feature"], shown["permutation_auc_drop"], color=BLUE)
        ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (test set)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show(fig)
        st.caption(
            "Importance shows what the model uses to predict, not what causes churn. Related features "
            "(for example age and age group) share importance."
        )


def performance_page():
    _, info = require_model()
    st.title("Model Performance")
    st.write(
        "All numbers come from the 2,000-customer test set that was kept unseen during training and "
        "model selection."
    )

    final = read_report("final_results.csv")
    if final is not None:
        st.subheader("Model comparison")
        final = final.rename(columns={final.columns[0]: "Model"})
        st.dataframe(final.round(3), hide_index=True)
        st.caption(
            "Each model uses the decision threshold chosen on training data (last column). The saved "
            "final model, Gradient Boosting, uses the rounded threshold of "
            f"{info['decision_threshold']:.2f}. Accuracy alone is not used to pick a model."
        )
    else:
        st.info("Run `python -m src.final_evaluation` to create the model comparison table.")

    metrics = info["metrics"]
    st.subheader(f"Final model: Gradient Boosting (threshold {info['decision_threshold']:.2f})")
    cols = st.columns(5)
    for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        col.metric(name, f"{metrics[name]:.3f}")

    preds = read_report("test_predictions.csv")
    if preds is None:
        st.info("Run `python -m src.train_model` to create the saved test predictions.")
        return
    y_true = preds["churn"].to_numpy()
    proba = preds["churn_probability"].to_numpy()

    threshold = st.slider(
        "Decision threshold (for exploring the precision-recall trade-off only)",
        0.05, 0.95, float(info["decision_threshold"]), 0.01,
    )
    pred = (proba >= threshold).astype(int)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(y_true, pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Stay", "Churn"])
        ax.set_yticks([0, 1], ["Stay", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        fig.tight_layout()
        show(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, color=RED, label=f"Gradient Boosting (AUC = {roc_auc_score(y_true, proba):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show(fig)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_true, pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_true, pred):.3f}")
    m4.metric("F1-score", f"{f1_score(y_true, pred):.3f}")
    m5.metric("Customers flagged", f"{pred.mean() * 100:.1f}%")
    st.caption(
        "Lowering the threshold catches more churners (higher recall) but flags more customers who would "
        "have stayed (lower precision). Moving this slider is for exploration; the official threshold is "
        f"{info['decision_threshold']:.2f}."
    )

    image = REPORT_DIR / "figures" / "23_final_roc_curves.png"
    with st.expander("ROC curves for all four models"):
        if image.exists():
            st.image(str(image))
        else:
            st.write("Run `python -m src.final_evaluation` to create this chart.")

    imbalance = read_report("imbalance_cv_results.csv")
    with st.expander("Class imbalance experiments (cross-validation on training data)"):
        if imbalance is not None:
            st.dataframe(imbalance.round(3), hide_index=True)
            st.caption("Baseline vs class weights vs SMOTE. ROC-AUC barely changes; recall rises while precision falls.")
        else:
            st.write("Run `python -m src.imbalance_experiments` to create this table.")

    baseline = read_report("baseline_results.csv")
    with st.expander("Baseline models at the default 0.5 threshold"):
        if baseline is not None:
            st.dataframe(baseline.round(3), hide_index=True)
            st.caption("The dummy model scores about 80% accuracy while finding no churners.")
        else:
            st.write("Run `python -m src.train_baseline` to create this table.")


PAGES = {
    "Dashboard": dashboard_page,
    "Churn Prediction": prediction_page,
    "Pattern Analysis": pattern_page,
    "Model Performance": performance_page,
}

st.sidebar.title("Churn Prediction")
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("Risk levels and thresholds are project-defined, not universal business standards.")
PAGES[choice]()