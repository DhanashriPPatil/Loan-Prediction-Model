import streamlit as st
import joblib  # Replacing pickle
import numpy as np

# Load the model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

# Header with background color
st.markdown('<div style="font-size: 40px; color: white; font-weight: bold; text-align: center; background-color: #3498db; padding: 20px; border-radius: 10px;">🏦 Loan Approval Prediction</div>', unsafe_allow_html=True)

# Form Inputs — with individual field-wise instructions
st.markdown("### Number of Dependents")
st.markdown("**Enter the total number of individuals financially dependent on you (children, parents, etc.).**")
no_of_dependents = st.number_input("Number of Dependents", min_value=0)

st.markdown("### Education")
st.markdown("**Enter your highest education level:** Graduate or Undergraduate.")
education_input = st.text_input("Education (Graduate/Undergraduate)")
education = 1 if education_input.strip().lower() == "graduate" else 0

st.markdown("### Self-Employed")
st.markdown("**Select 'Yes' if you run your own business or freelance, otherwise select 'No'.**")
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

st.markdown("### Annual Income (₹)")
st.markdown("**Enter your total yearly income from all sources (salary, business, investments, etc.).**")
income_annum = st.number_input("Annual Income (₹)", min_value=0)

st.markdown("### Loan Amount (₹)")
st.markdown("**Enter the loan amount you wish to borrow (in Indian Rupees).**")
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)

st.markdown("### Loan Term (in months)")
st.markdown("**Enter the loan repayment duration in months (e.g., 120 months = 10 years).**")
loan_term = st.number_input("Loan Term (Months)", min_value=0)

st.markdown("### CIBIL Score")
st.markdown("**Provide your CIBIL credit score (must be between 300 and 900).**")
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

st.markdown("### Residential Assets Value (₹)")
st.markdown("**Enter the total value of your owned residential properties (houses, apartments).**")
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0)

st.markdown("### Commercial Assets Value (₹)")
st.markdown("**Enter the total value of your owned commercial properties (shops, warehouses).**")
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0)

st.markdown("### Luxury Assets Value (₹)")
st.markdown("**Enter the total value of your owned luxury items (luxury cars, jewelry, etc.).**")
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0)

st.markdown("### Bank Asset Value (₹)")
st.markdown("**Enter the total value of your assets held in banks (savings accounts, fixed deposits, etc.).**")
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)

# Convert categorical values to numeric
self_employed = 1 if self_employed == "Yes" else 0

# Predict button
if st.button("🔍 Predict Loan Approval"):
    input_data = np.array([[no_of_dependents, education, self_employed, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value]])

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    # Display prediction result
    if prediction[0] == 1:
        st.success("✅ Your loan is likely to be Approved!")
    else:
        st.error("❌ Your loan is likely to be Rejected.")
