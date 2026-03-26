import streamlit as st
import joblib
import numpy as np


model = joblib.load("V:/PROJECTS/MINIOR2/code/credit_model.pkl")



st.title("🏦 Loan Approval Prediction App")

st.write("Enter applicant details to predict loan approval")


age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
income = st.number_input("Annual Income", min_value=0, value=50000)
education = st.selectbox("Education Level (Encoded)", [0, 1, 2, 3])
home_ownership = st.selectbox("Home Ownership (Encoded)", [0, 1, 2, 3])
employment_length = st.number_input("Employment Length (Years)", min_value=0, value=5)
interest_rate = st.number_input("Loan Interest Rate (%)", value=10.0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)


if st.button("Predict Loan Status"):
    input_data = np.array([[age, gender, income, education,
                            home_ownership, employment_length,
                            interest_rate, loan_amount, credit_score]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
