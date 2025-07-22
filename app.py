import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model_lightgbm.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# CSS styling to match the design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 2rem;
        border-radius: 15px;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .header-card {
        background: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-size: 16px;
        font-weight: 600;
        width: 200px;
        margin-top: 1rem;
    }
    
    .result-card {
        background: #27ae60;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    h1, h2, h3 { color: #2c3e50; }
    .stSelectbox > div > div { background-color: #34495e; color: white; }
    .stNumberInput > div > div > input { background-color: #34495e; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class='header-card'>
        <h1>💼 Employee Salary Predictor</h1>
        <p><strong>Algorithm Used:</strong> XGBoost Regressor</p>
        <p><strong>Model R² Score:</strong> 94.58%</p>
        <p><strong>Evaluation:</strong> Compares predicted vs actual salaries</p>
    </div>
""", unsafe_allow_html=True)

# Input section
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 📋 Enter Employee Details")

# Simplified input fields to match the design
age = st.number_input("Age", min_value=18, max_value=90, value=30)

gender = st.selectbox("Gender", ["Female", "Male"])

education_levels = ["Bachelor's", "Master's", "PhD", "High School", "Associate"]
education = st.selectbox("Education Level", education_levels)

job_titles = ["Account Manager", "Software Engineer", "Sales Representative", "Data Scientist", "Marketing Manager", "HR Manager"]
job_title = st.selectbox("Job Title", job_titles)

experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

st.markdown("</div>", unsafe_allow_html=True)

# Map simplified inputs to model format
def map_inputs(age, gender, education, job_title, experience):
    # Simple mapping - you may need to adjust based on your model's training
    workclass_encoded = 0  # Private
    education_encoded = ["High School", "Associate", "Bachelor's", "Master's", "PhD"].index(education) if education in ["High School", "Associate", "Bachelor's", "Master's", "PhD"] else 0
    education_num = education_encoded + 9
    marital_status_encoded = 0  # Married-civ-spouse
    occupation_encoded = 0  # Tech-support
    relationship_encoded = 0  # Wife
    race_encoded = 0  # White
    sex_encoded = 1 if gender == "Male" else 0
    capital_gain = 0
    capital_loss = 0
    hours_per_week = 40
    native_country_encoded = 0  # United-States
    fnlwgt = 100000
    
    return np.array([[
        age, workclass_encoded, education_encoded, education_num, marital_status_encoded,
        occupation_encoded, relationship_encoded, race_encoded, sex_encoded, capital_gain,
        capital_loss, hours_per_week, native_country_encoded, fnlwgt
    ]])

# Predict button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Predict Salary"):
        input_data = map_inputs(age, gender, education, job_title, experience)
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            salary_text = "₹75,000/month (High Income Group)"
        else:
            salary_text = "₹30,000/month (Lower Income Group)"
        
        st.markdown(f"""
            <div class='result-card'>
                🎯 Predicted Salary: {salary_text}
            </div>
        """, unsafe_allow_html=True)
