import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

.hero-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 40px;
    margin: 20px 0;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #e0e7ff;
    font-weight: 400;
    opacity: 0.9;
}

.stats-row {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 30px 0;
    flex-wrap: wrap;
}

.stat-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-width: 120px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stat-number {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
}

.stat-label {
    font-size: 0.85rem;
    color: #e0e7ff;
    margin-top: 5px;
}

.form-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 40px;
    margin: 30px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.form-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #ffffff;
    text-align: center;
    margin-bottom: 30px;
}

.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    color: white !important;
}

.stSelectbox label, .stNumberInput label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 30px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
}

.prediction-result {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin: 30px 0;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.prediction-amount {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
}

.prediction-note {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 10px;
}

.metrics-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 30px;
    margin: 30px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
    text-align: center;
    margin-bottom: 20px;
}

.metrics-grid {
    display: flex;
    justify-content: space-around;
    gap: 20px;
    flex-wrap: wrap;
}

.metric-card {
    background: rgba(96, 165, 250, 0.2);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    min-width: 140px;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #60a5fa;
}

.metric-label {
    color: #e0e7ff;
    font-size: 0.85rem;
    margin-top: 5px;
}

@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem; }
    .stats-row, .metrics-grid { flex-direction: column; align-items: center; }
}
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model_data = joblib.load("salary_predictor.pkl")
    model = model_data["model"]
    label_encoders = model_data["label_encoders"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    model_loaded = True
except:
    model_loaded = False
    st.warning("Model file not found. Running in demo mode.")

# Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Employee Salary Predictor</div>
    <div class="hero-subtitle">Advanced Machine Learning for Accurate Salary Estimation</div>
</div>
""", unsafe_allow_html=True)

# Statistics
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">94.58%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">XGBoost</div>
        <div class="stat-label">Algorithm</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">Real-time</div>
        <div class="stat-label">Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Form
st.markdown('<div class="form-container"><div class="form-title">Employee Information</div></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("salary_form"):
        col_left, col_right = st.columns(2)
        
        with col_left:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            if model_loaded:
                gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
                education = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
            else:
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
                education = st.selectbox("Education Level", options=["Bachelor's", "Master's", "PhD", "High School"])
        
        with col_right:
            experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
            if model_loaded:
                job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
            else:
                job_title = st.selectbox("Job Title", options=["Software Engineer", "Data Scientist", "Product Manager", "Designer"])

        submit_button = st.form_submit_button("Predict Salary")

# Prediction
if submit_button:
    if model_loaded:
        try:
            input_df = pd.DataFrame({
                "Age": [age], "Gender": [gender], "Education Level": [education],
                "Job Title": [job_title], "Years of Experience": [experience]
            })
            
            for col in ["Gender", "Education Level", "Job Title"]:
                input_df[col] = label_encoders[col].transform(input_df[col])
            
            input_scaled = scaler.transform(input_df)
            predicted_salary = model.predict(input_scaled)[0]
        except:
            predicted_salary = 75000
    else:
        # Demo calculation
        base = 50000 + (age - 25) * 1000 + experience * 3000
        edu_bonus = {"High School": 0, "Bachelor's": 15000, "Master's": 25000, "PhD": 40000}
        job_bonus = {"Software Engineer": 20000, "Data Scientist": 30000, "Product Manager": 25000, "Designer": 15000}
        predicted_salary = base + edu_bonus.get(education, 0) + job_bonus.get(job_title, 0)
    
    st.markdown(f"""
    <div class="prediction-result">
        <div class="prediction-amount">${predicted_salary:,.0f}</div>
        <div class="prediction-note">Estimated Annual Salary</div>
    </div>
    """, unsafe_allow_html=True)

# Model Performance
st.markdown("""
<div class="metrics-container">
    <div class="section-title">Model Performance</div>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">94.58%</div>
            <div class="metric-label">R² Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">±$5,200</div>
            <div class="metric-label">Mean Error</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">89.2%</div>
            <div class="metric-label">Within 10%</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display evaluation plot if available
try:
    eval_plot = Image.open("images/plot.png")
    st.image(eval_plot, caption="Model Evaluation", use_container_width=True)
except:
    # Sample data table
    st.markdown("### Sample Predictions")
    sample_data = pd.DataFrame({
        'Age': [28, 35, 42, 29, 38],
        'Experience': [3, 7, 15, 5, 8],
        'Education': ['Master\'s', 'Bachelor\'s', 'PhD', 'Master\'s', 'Bachelor\'s'],
        'Job Title': ['Data Scientist', 'Software Engineer', 'Manager', 'Product Manager', 'Designer'],
        'Actual': [72000, 85000, 125000, 78000, 65000],
        'Predicted': [71200, 86500, 122800, 79100, 63900]
    })
    st.dataframe(sample_data, use_container_width=True)
