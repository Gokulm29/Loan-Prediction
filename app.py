# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bank Loan Approval Predictor", page_icon="💳", layout="centered")

DATA_SCHEMA_HINT = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": "Number (0,1,2,3+)",
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "ApplicantIncome": "Number",
    "CoapplicantIncome": "Number",
    "LoanAmount": "Number (in thousands)",
    "Loan_Amount_Term": "Term in months (e.g., 360)",
    "Credit_History": "1.0 = good, 0.0 = bad",
    "Property_Area": ["Urban", "Semiurban", "Rural"]
}

# Load model
MODEL_PATH = "loan_approval_model.pkl"
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
feature_columns = bundle["feature_columns"]

st.title("💳 Bank Loan Approval Predictor")
st.write("Enter applicant details to predict **Loan Approval (Yes/No)**.")

# ---- UI controls (adjust based on your dataset columns) ----
def number_input(label, min_val=0.0, step=1.0, value=None):
    return st.number_input(label, min_value=float(min_val), step=float(step), value=float(value) if value is not None else None)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self_Employed", ["Yes", "No"])
    property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])

with col2:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    applicant_income = st.number_input("ApplicantIncome", min_value=0, step=100, value=5000)
    coapplicant_income = st.number_input("CoapplicantIncome", min_value=0, step=100, value=0)
    loan_amount = st.number_input("LoanAmount (in thousands)", min_value=0, step=1, value=150)
    loan_term = st.number_input("Loan_Amount_Term (months)", min_value=12, step=12, value=360)
    credit_history = st.selectbox("Credit_History", [1.0, 0.0])

# Build a single-row DataFrame in *training feature* order; missing columns will be added if needed
input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_emp,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

# Some datasets may not have all of the above or may have extra columns.
# We'll create a frame with exactly the columns used during training.
row = pd.DataFrame([input_dict])

# Add any missing training columns with NaN; drop any extras
for c in feature_columns:
    if c not in row.columns:
        row[c] = np.nan
row = row[feature_columns]

st.divider()
if st.button("Predict"):
    pred = pipe.predict(row)[0]
    proba = getattr(pipe.named_steps["model"], "predict_proba", lambda X: [[None, None]])(pipe[:-1].transform(row))
    # Handle models without predict_proba
    prob_approved = None
    if isinstance(proba, (list, np.ndarray)):
        try:
            prob_approved = float(proba[0][1])
        except Exception:
            prob_approved = None

    if pred == 1:
        st.success(f"✅ Prediction: **Approved**" + (f" (confidence ~ {prob_approved:.2%})" if prob_approved is not None else ""))
    else:
        st.error(f"❌ Prediction: **Not Approved**" + (f" (confidence ~ {prob_approved:.2%})" if prob_approved is not None else ""))

st.caption("Note: This is a demo ML model. Use responsibly.")
