import warnings
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from src.config import FEATURE_INFO_PATH, MODEL_PATH
from src.preprocessing import add_engineered_features


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    info = joblib.load(FEATURE_INFO_PATH)
    if info["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"Model was saved with scikit-learn {info['sklearn_version']} but "
            f"{sklearn.__version__} is installed. Re-run src/train_model.py if you see errors."
        )
    return model, info


def get_risk_level(probability, info):
    cutoffs = info["risk_cutoffs"]
    if probability < cutoffs["low_max"]:
        return "Low"
    if probability < cutoffs["high_min"]:
        return "Medium"
    return "High"


def _input_frame(rows, info):
    frame = pd.DataFrame(rows, columns=info["input_columns"])
    return add_engineered_features(frame)[info["model_columns"]]


def explain_customer(model, info, customer):
    base = dict(customer)
    variants = [base]
    for col in info["input_columns"]:
        changed = dict(base)
        changed[col] = info["reference_values"][col]
        variants.append(changed)
    probabilities = model.predict_proba(_input_frame(variants, info))[:, 1]
    factors = []
    for col, p_ref in zip(info["input_columns"], probabilities[1:]):
        effect = (probabilities[0] - p_ref) * 100
        factors.append({
            "feature": col,
            "value": base[col],
            "typical_value": info["reference_values"][col],
            "effect_pct_points": round(float(effect), 1),
            "direction": "raises risk" if effect > 0 else "lowers risk",
        })
    return sorted(factors, key=lambda f: abs(f["effect_pct_points"]), reverse=True)


def predict_customer(customer, top_n=4, min_effect=1.0):
    model, info = load_artifacts()
    missing = [c for c in info["input_columns"] if c not in customer]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    row = _input_frame([{c: customer[c] for c in info["input_columns"]}], info)
    probability = float(model.predict_proba(row)[0, 1])
    factors = explain_customer(model, info, {c: customer[c] for c in info["input_columns"]})
    return {
        "probability": probability,
        "probability_pct": round(probability * 100, 1),
        "prediction": "Likely to Churn" if probability >= info["decision_threshold"] else "Likely to Stay",
        "risk_level": get_risk_level(probability, info),
        "top_factors": [f for f in factors if abs(f["effect_pct_points"]) >= min_effect][:top_n],
    }


if __name__ == "__main__":
    examples = {
        "Customer A": dict(credit_score=600, country="Germany", gender="Female", age=52, tenure=3,
                           balance=125000, products_number=1, credit_card=1, active_member=0,
                           estimated_salary=90000),
        "Customer B": dict(credit_score=720, country="France", gender="Male", age=29, tenure=6,
                           balance=0, products_number=2, credit_card=1, active_member=1,
                           estimated_salary=60000),
        "Customer C": dict(credit_score=650, country="Spain", gender="Female", age=45, tenure=5,
                           balance=80000, products_number=1, credit_card=0, active_member=1,
                           estimated_salary=70000),
    }
    for name, customer in examples.items():
        result = predict_customer(customer)
        print(f"\n{name}: {customer}")
        print(f"  Churn Probability: {result['probability_pct']}%")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Risk Level: {result['risk_level']}")
        for f in result["top_factors"]:
            print(f"    {f['feature']} = {f['value']} (typical {f['typical_value']}): "
                  f"{f['effect_pct_points']:+.1f} points, {f['direction']}")