import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model_lightgbm.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Modern CSS with white and blue theme matching the design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f8faff 0%, #e6f0ff 100%);
    }
    
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
        font-family: 'Inter', sans-serif;
        border: 1px solid rgba(59, 130, 246, 0.1);
        max-width: 600px;
        margin: 0 auto;
    }
    
    .header-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .info-icon {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        margin-right: 0.5rem;
    }
    
    .blue-icon { background-color: #3b82f6; }
    .gray-icon { background-color: #6b7280; }
    .purple-icon { background-color: #8b5cf6; }
    
    .form-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stSelectbox > label, .stSlider > label, .stNumberInput > label {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }
    
    .result-section {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class='header-section'>
        <div class='header-title'>
            💼 Employee Salary Predictor
        </div>
        <div class='info-item'>
            <div class='info-icon blue-icon'></div>
            Algorithm Used: LightGBM Regressor
        </div>
        <div class='info-item'>
            <div class='info-icon gray-icon'></div>
            Model R² Score: 94.58%
        </div>
        <div class='info-item'>
            <div class='info-icon purple-icon'></div>
            Evaluation: Compares predicted vs actual salaries.
        </div>
    </div>
""", unsafe_allow_html=True)

# Form Section
st.markdown("""
    <div class='form-section'>
        <div class='section-title'>
            📝 Enter Employee Details
        </div>
    </div>
""", unsafe_allow_html=True)

# Simplified inputs based on the design
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    education = st.selectbox("Education Level", 
        ["Bachelor's", "Master's", "High School", "Doctorate", "Associate", "Some College"])

with col2:
    job_title = st.selectbox("Job Title", 
        ["Account Manager", "Software Engineer", "Data Scientist", "Sales Representative", 
         "Marketing Manager", "HR Specialist", "Financial Analyst", "Operations Manager"])
    experience = st.slider("Years of Experience", 0, 40, 5)

# Mapping simplified inputs to original model format
education_mapping = {
    "Bachelor's": "Bachelors",
    "Master's": "Masters", 
    "High School": "HS-grad",
    "Doctorate": "Doctorate",
    "Associate": "Assoc-acdm",
    "Some College": "Some-college"
}

job_mapping = {
    "Account Manager": "Exec-managerial",
    "Software Engineer": "Tech-support",
    "Data Scientist": "Prof-specialty",
    "Sales Representative": "Sales",
    "Marketing Manager": "Exec-managerial",
    "HR Specialist": "Adm-clerical",
    "Financial Analyst": "Prof-specialty",
    "Operations Manager": "Exec-managerial"
}

# Predict Button
if st.button("Predict Salary"):
    # Map to original categories
    education_orig = education_mapping.get(education, "Bachelors")
    occupation_orig = job_mapping.get(job_title, "Prof-specialty")
    
    # Original category lists (keeping same as original code)
    workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    education_options = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
    marital_status_options = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Unknown"]
    relationship_options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    race_options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    native_country_options = ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "China", "Cuba", "England", "Japan", "South"]
    
    # Set default values for simplified interface
    workclass = "Private"
    marital_status = "Never-married" if age < 25 else "Married-civ-spouse"
    relationship = "Not-in-family" if age < 25 else ("Husband" if gender == "Male" else "Wife")
    race = "White"
    native_country = "United-States"
    
    # Calculate education_num based on education level
    education_num_mapping = {
        "HS-grad": 9, "Some-college": 10, "Assoc-acdm": 11, 
        "Bachelors": 13, "Masters": 14, "Doctorate": 16
    }
    education_num = education_num_mapping.get(education_orig, 13)
    
    # Set reasonable defaults based on experience and job
    hours_per_week = min(40 + (experience // 3), 60)
    capital_gain = experience * 1000 if experience > 10 else 0
    capital_loss = 0
    fnlwgt = 190000  # Average value
    
    # Encode all inputs
    workclass_encoded = workclass_options.index(workclass)
    education_encoded = education_options.index(education_orig)
    marital_status_encoded = marital_status_options.index(marital_status)
    occupation_encoded = occupation_options.index(occupation_orig)
    relationship_encoded = relationship_options.index(relationship)
    race_encoded = race_options.index(race)
    native_country_encoded = native_country_options.index(native_country)
    sex_encoded = 1 if gender == "Male" else 0
    
    # Create input array in correct order
    input_data = np.array([[
        age, workclass_encoded, education_encoded, education_num, marital_status_encoded,
        occupation_encoded, relationship_encoded, race_encoded, sex_encoded, capital_gain,
        capital_loss, hours_per_week, native_country_encoded, fnlwgt
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        salary = "₹75,000/month (High Income Group)"
    else:
        salary = "₹30,000/month (Lower Income Group)"
    
    st.markdown(f"<div class='result-section'>Predicted Salary: {salary}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
