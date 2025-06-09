import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.pkl")

pipeline = load_model()

st.title("🔍 Customer Churn Prediction")

# Collect all required features
gender = st.selectbox("Gender", ["Female", "Male"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education = st.selectbox("Education Level", ["Bachelor", "High School", "PhD", "Master"])
occupation = st.selectbox("Occupation", ["Employed", "Unemployed", "Self-Employed", "Student", "Retired"])
customer_segment = st.selectbox("Customer Segment", ["Standard", "Premium", "Basic"])
communication_channel = st.selectbox("Preferred Communication Channel", ["Email", "SMS", "Phone"])


credit_score = st.slider("Credit Score", 300, 850, 650)
income = st.number_input("Annual Income", 10000, 500000, 50000)
balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
outstanding_loans = st.number_input("Outstanding Loans", 0.0, 100000.0, 20000.0)
credit_history_length = st.slider("Credit History Length (years)", 0, 30, 5)
customer_tenure = st.slider("Customer Tenure (years)", 0, 20, 3)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
num_complaints = st.slider("Number of Complaints", 0, 10, 0)
dependents = st.slider("Number of Dependents", 0, 10, 0)

# Map into DataFrame
input_df = pd.DataFrame({
    "Gender": [gender],
    "MaritalStatus": [marital_status],
    "EducationLevel": [education],
    "Occupation": [occupation],
    "CustomerSegment": [customer_segment],
    "PreferredCommunicationChannel": [communication_channel],
    "CreditScore": [credit_score],
    "Income": [income],
    "Balance": [balance],
    "OutstandingLoans": [outstanding_loans],
    "CreditHistoryLength": [credit_history_length],
    "CustomerTenure": [customer_tenure],
    "NumOfProducts": [num_products],
    "NumComplaints": [num_complaints],
    "NumberofDependents": [dependents]
})

# Predict
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.markdown(
            """
            <div style='background-color:#ffcccc;padding:20px;border-radius:10px'>
                <h3 style='color:#990000'>🚨 Prediction: Customer Will Churn</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='background-color:#ccffcc;padding:20px;border-radius:10px'>
                <h3 style='color:#006600'>✅ Prediction: Customer Will Not Churn</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
