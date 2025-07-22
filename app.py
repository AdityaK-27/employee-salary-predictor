import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model_lightgbm.pkl")

st.set_page_config(page_title="Employee Salary Estimator", layout="centered")

# Enhanced CSS with white background and professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        font-family: 'Inter', sans-serif;
        border: 1px solid #e1e5e9;
    }
    
    h1 {
        color: #1e293b;
        text-align: center;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        height: 3.2em;
        width: 100%;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 500;
        text-align: center;
        border: none;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .chart-title {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 1.5rem 0 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("💼 Employee Salary Estimator")
st.markdown("<p class='subtitle'>Professional salary estimation using advanced machine learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Actual category labels (unchanged from original)
workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education_options = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"]
marital_status_options = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Unknown"]
relationship_options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race_options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
native_country_options = ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "China", "Cuba", "England", "Japan", "South"]

# Collect user inputs (unchanged layout, just organized in columns for better presentation)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", workclass_options)
    education = st.selectbox("Education", education_options)
    education_num = st.slider("Education Number", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    occupation = st.selectbox("Occupation", occupation_options)
    relationship = st.selectbox("Relationship", relationship_options)

with col2:
    race = st.selectbox("Race", race_options)
    sex = st.selectbox("Sex", ["Female", "Male"])
    capital_gain = st.number_input("Capital Gain", 0)
    capital_loss = st.number_input("Capital Loss", 0)
    hours_per_week = st.slider("Hours Per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", native_country_options)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 0)

# Encode inputs (unchanged from original)
workclass_encoded = workclass_options.index(workclass)
education_encoded = education_options.index(education)
marital_status_encoded = marital_status_options.index(marital_status)
occupation_encoded = occupation_options.index(occupation)
relationship_encoded = relationship_options.index(relationship)
race_encoded = race_options.index(race)
native_country_encoded = native_country_options.index(native_country)
sex_encoded = 1 if sex == "Male" else 0

# Input feature order must match model training (unchanged from original)
input_data = np.array([[
    age, workclass_encoded, education_encoded, education_num, marital_status_encoded,
    occupation_encoded, relationship_encoded, race_encoded, sex_encoded, capital_gain,
    capital_loss, hours_per_week, native_country_encoded, fnlwgt
]])

# Predict and estimate salary (unchanged logic, enhanced presentation)
if st.button("Estimate Salary"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        salary = "₹75,000/month (High Income Group)"
        salary_value = 75000
    else:
        salary = "₹30,000/month (Lower Income Group)"
        salary_value = 30000

    st.success(f"Estimated Salary: {salary}")

    # Enhanced salary bar chart
    st.markdown("<p class='chart-title'>📊 Salary Estimate Visualization</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Predicted Salary"], [salary_value], color="#3b82f6", alpha=0.8, width=0.5)
    ax.set_ylabel("Amount (INR)", fontweight='500')
    ax.set_ylim(0, 100000)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value label on bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2000,
                f'₹{height:,}', ha='center', va='bottom', fontweight='500')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
