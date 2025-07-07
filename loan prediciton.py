import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Predictor", layout="centered")

# --- Light Theme Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdfcfb, #e2d1c3);
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #4b3832;
    }
    label, .stSelectbox, .stNumberInput, .stSlider {
        color: black !important;
    }
    .stButton>button {
        background-color: #ffcf40;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #fff2b2;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Loan Eligibility & EMI Predictor")

# --- Training Data & Model ---
data = pd.DataFrame({
    "Gender": [1, 0, 1, 0, 1],
    "Married": [1, 0, 1, 1, 0],
    "Education": [1, 1, 0, 1, 1],
    "ApplicantIncome": [5000, 3000, 4000, 6000, 3500],
    "LoanAmount": [200, 100, 150, 250, 120],
    "Credit_History": [1, 1, 1, 0, 1],
    "Loan_Status": [1, 1, 0, 0, 1]
})

X = data[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = data['Loan_Status']
model = LogisticRegression()
model.fit(X, y)

# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Marital Status", ["Yes", "No"])
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
with col2:
    income = st.number_input("💼 Monthly Income (₹)", min_value=1000, step=500, value=3000)
    loan = st.number_input("💳 Loan Amount (₹ in 1000s)", min_value=50, step=10, value=100)
    credit = st.selectbox("📊 Credit History", [1, 0])

interest = st.slider("📈 Interest Rate (Annual %)", 5.0, 20.0, 10.0)
years = st.slider("📆 Loan Tenure (Years)", 1, 20, 5)

# --- Predict & Show Result ---
if st.button("Check Loan & EMI"):
    input_data = np.array([
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        1 if education == "Graduate" else 0,
        income,
        loan,
        credit
    ]).reshape(1, -1)

    result = model.predict(input_data)[0]

    st.subheader("🔍 Result:")
    if result == 1:
        st.success("✅ Loan Approved")

        # EMI Calculation
        P = loan * 1000
        r = interest / 12 / 100
        n = years * 12
        emi = (P * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
        total = emi * n

        st.write(f"💰 **Monthly EMI:** ₹{emi:,.2f}")
        st.write(f"📉 **Total Repayable:** ₹{total:,.2f}")
    else:
        st.error("❌ Loan Rejected")