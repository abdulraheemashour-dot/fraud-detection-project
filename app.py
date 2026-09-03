import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

# ---------- load model & preprocessor ----------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")

    models = {}
    model_files = {
        "Decision Tree":  "decision_tree_model.pkl",
        "Random Forest":  "random_forest_model.pkl",
        "Linear SVM":     "linear_svc_fraud_model.pkl",
        "Naive Bayes":    "naive_bayes_model.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    return preprocessor, models

preprocessor, models = load_artifacts()

# ---------- UI ----------
st.title("🔍 Insurance Fraud Detection")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox("Model", list(models.keys()))

    country = st.selectbox("Country", [
        "Côte d'Ivoire", "Ethiopia", "Ghana", "Kenya",
        "Mozambique", "Nigeria", "Rwanda", "Senegal",
        "South Africa", "Tanzania", "Uganda", "Zambia"
    ])

    claim_type = st.selectbox("Claim Type", [
        "crop", "health", "life", "motor", "property", "travel"
    ])

    policy_duration_months = st.number_input("Policy Duration (months)", min_value=1, max_value=240, value=24)
    previous_claims_count  = st.number_input("Previous Claims Count",     min_value=0, max_value=20,  value=0)

with col2:
    reporting_delay_days        = st.number_input("Reporting Delay (days)",         min_value=0,   max_value=365, value=5)
    document_completeness_score = st.slider(      "Document Completeness Score",     min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    claimant_age                = st.number_input("Claimant Age",                    min_value=18,  max_value=100, value=35)
    seasonality_flag            = st.selectbox(   "Seasonality Flag",               [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    amount_deviation_zscore     = st.number_input("Amount Deviation (Z-score)",      min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
    duplicate_claim_flag        = st.selectbox(   "Duplicate Claim Flag",           [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

st.markdown("---")
predict_btn = st.button("Predict", use_container_width=True, type="primary")

if predict_btn:
    input_df = pd.DataFrame([{
        "policy_duration_months":       policy_duration_months,
        "previous_claims_count":        previous_claims_count,
        "reporting_delay_days":         reporting_delay_days,
        "document_completeness_score":  document_completeness_score,
        "claimant_age":                 claimant_age,
        "seasonality_flag":             seasonality_flag,
        "amount_deviation_zscore":      amount_deviation_zscore,
        "duplicate_claim_flag":         duplicate_claim_flag,
        "country":                      country,
        "claim_type":                   claim_type,
    }])

    processed   = preprocessor.transform(input_df)
    model       = models[selected_model]
    prediction  = model.predict(processed)[0]

    # probability — LinearSVC uses decision_function
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(processed)[0][1]
        prob_text = f"{prob * 100:.1f}%"
    elif hasattr(model, "decision_function"):
        score = model.decision_function(processed)[0]
        prob_text = f"Score: {score:.3f}"
    else:
        prob_text = "N/A"

    st.markdown("### Result")

    if prediction == 1:
        st.error(f"⚠️ **FRAUD DETECTED**   |   Confidence: {prob_text}")
    else:
        st.success(f"✅ **LEGITIMATE CLAIM**   |   Confidence: {prob_text}")

    with st.expander("Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
