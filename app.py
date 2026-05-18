import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/xgboost_model.pkl")

st.title("Credit Risk Prediction App")

st.write("Enter borrower details to predict default risk")

# -------------------
# INPUTS
# -------------------
person_age = st.number_input("Age", 18, 100, 30)
person_income = st.number_input("Income", 1000, 1000000, 50000)
person_emp_length = st.number_input("Employment Length", 0, 50, 5)

loan_grade = st.selectbox("Loan Grade (0=best)", [0,1,2,3,4,5,6])
loan_amnt = st.number_input("Loan Amount", 1000, 50000, 10000)
loan_int_rate = st.number_input("Interest Rate", 0.0, 30.0, 10.0)
loan_percent_income = st.number_input("Loan % Income", 0.0, 1.0, 0.2)

cb_default = st.selectbox("Previous Default (0/1)", [0,1])
cred_hist = st.number_input("Credit History Length", 0, 50, 5)

home = st.selectbox("Home Ownership", ["OTHER", "OWN", "RENT"])
intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

# -------------------
# ONE-HOT ENCODING (must match training)
# -------------------
home_other = 1 if home == "OTHER" else 0
home_own = 1 if home == "OWN" else 0
home_rent = 1 if home == "RENT" else 0

intent_edu = 1 if intent == "EDUCATION" else 0
intent_home = 1 if intent == "HOMEIMPROVEMENT" else 0
intent_med = 1 if intent == "MEDICAL" else 0
intent_personal = 1 if intent == "PERSONAL" else 0
intent_venture = 1 if intent == "VENTURE" else 0

# -------------------
# PREDICTION
# -------------------
if st.button("Predict Risk"):

    input_data = pd.DataFrame([[
        person_age,
        person_income,
        person_emp_length,
        loan_grade,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_default,
        cred_hist,
        home_other,
        home_own,
        home_rent,
        intent_edu,
        intent_home,
        intent_med,
        intent_personal,
        intent_venture
    ]], columns=[
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "person_home_ownership_OTHER",
        "person_home_ownership_OWN",
        "person_home_ownership_RENT",
        "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_PERSONAL",
        "loan_intent_VENTURE"
    ])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.write("### Result")

    if pred == 1:
        st.error("Prediction: DEFAULT")
    else:
        st.success("Prediction: NO DEFAULT")

    st.write("Default Probability:", round(proba, 3))

    # Risk level
    if proba < 0.3:
        st.success("Low Risk")
    elif proba < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")