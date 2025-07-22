import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model_lightgbm_v2.pkl")

st.set_page_config(page_title="Employee Salary Estimator", layout="centered")

# Custom CSS to style the app
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("💼 Employee Salary Estimator")
st.write("Use the form below to estimate an employee's expected monthly salary.")
st.markdown("---")

# Actual category labels
workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education_options = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
marital_status_options = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Unknown"]
relationship_options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race_options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
native_country_options = ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "China", "Cuba", "England", "Japan", "South"]

# Collect user inputs
age = st.slider("Age", 18, 90, 30)
workclass = st.selectbox("Workclass", workclass_options)
education = st.selectbox("Education", education_options)
education_num = st.slider("Education Number", 1, 16, 9)
marital_status = st.selectbox("Marital Status", marital_status_options)
occupation = st.selectbox("Occupation", occupation_options)
relationship = st.selectbox("Relationship", relationship_options)
race = st.selectbox("Race", race_options)
sex = st.selectbox("Sex", ["Female", "Male"])
capital_gain = st.number_input("Capital Gain", 0)
capital_loss = st.number_input("Capital Loss", 0)
hours_per_week = st.slider("Hours Per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", native_country_options)
fnlwgt = st.number_input("Final Weight (fnlwgt)", 0)

# Encode inputs
workclass_encoded = workclass_options.index(workclass)
education_encoded = education_options.index(education)
marital_status_encoded = marital_status_options.index(marital_status)
occupation_encoded = occupation_options.index(occupation)
relationship_encoded = relationship_options.index(relationship)
race_encoded = race_options.index(race)
native_country_encoded = native_country_options.index(native_country)
sex_encoded = 1 if sex == "Male" else 0

# Input feature order must match model training
input_data = np.array([[
    age, workclass_encoded, education_encoded, education_num, marital_status_encoded,
    occupation_encoded, relationship_encoded, race_encoded, sex_encoded, capital_gain,
    capital_loss, hours_per_week, native_country_encoded, fnlwgt
]])

# Predict and estimate salary
if st.button("Estimate Salary"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        salary = "₹75,000/month (High Income Group)"
        salary_value = 75000
    else:
        salary = "₹30,000/month (Lower Income Group)"
        salary_value = 30000

    st.success(f"Estimated Salary: {salary}")

    # Salary bar chart
    st.write("### Salary Estimate Chart")
    fig, ax = plt.subplots()
    ax.bar(["Predicted Salary"], [salary_value], color="#1f77b4")
    ax.set_ylabel("INR")
    ax.set_ylim(0, 100000)
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
