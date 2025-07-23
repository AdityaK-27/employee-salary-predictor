import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("salary_prediction_model.pkl")
model_columns = joblib.load("model_columns.pkl")
X_test, y_test, y_pred = joblib.load("test_predictions.pkl")
original_data = pd.read_csv("Salary Data.csv")
original_data.dropna(inplace=True)

# --- Streamlit UI Config ---
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stButton > button {
        background-color: #0d6efd;
        color: white;
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 15px;
    }
    .stButton > button:hover {
        background-color: #0b5ed7;
    }
    .stRadio label, .stSelectbox label, .stSlider label {
        font-size: 16px;
    }
    .stExpanderHeader {
        font-size: 16px;
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Employee Salary Predictor")
    
    # Model performance metrics first
    st.markdown("---")
    st.subheader("Model Performance")
    st.write(f"**Model:** `{model.__class__.__name__}`")
    st.metric("MAE", f"{abs(y_test - y_pred).mean():,.2f}")
    st.metric("MSE", f"{(y_test - y_pred).pow(2).mean():,.2f}")
    st.metric("R²", f"{model.score(X_test, y_test):.4f}")

    
    st.markdown("---")
    # Model info
    st.markdown("""
    This application estimates employee salaries based on:
    - Age  
    - Gender  
    - Education Level  
    - Job Title  
    - Years of Experience  
    """)


# --- Title ---
st.markdown("<h1 style='text-align:center;'>Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6c757d;'>Estimate salaries using a trained Regression Model</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #dee2e6;'>", unsafe_allow_html=True)

# --- Form for Input ---
st.header("Enter Employee Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])

with col2:
    job = st.selectbox("Job Title", original_data["Job Title"].unique())
    exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)


# --- Validations ---
if exp >= age or exp > (age - 20) or age < (exp + 18):
    st.error("Please enter valid age and experience values.")
    st.stop()
if education == "Master's" and age < 23:
    st.error("Age too low for a Master's degree.")
    st.stop()
if education == "PhD" and age < 26:
    st.error("Age too low for a PhD.")
    st.stop()

# --- Build Input Vector ---
input_dict = {col: 0 for col in model_columns}
input_dict["Age"] = age
input_dict["Years of Experience"] = exp

if f"Gender_{gender}" in input_dict:
    input_dict[f"Gender_{gender}"] = 1
if f"Education Level_{education}" in input_dict:
    input_dict[f"Education Level_{education}"] = 1
if f"Job Title_{job}" in input_dict:
    input_dict[f"Job Title_{job}"] = 1

X_input = pd.DataFrame([input_dict])[model_columns]

# --- Prediction Button ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Salary"):
    salary = model.predict(X_input)[0]
    st.success(f"Estimated Salary: ₹{salary:,.2f}")
    st.markdown("<hr style='border: 1px solid #dee2e6;'>", unsafe_allow_html=True)

# --- Feature Importance Chart ---
with st.expander("Feature Importance"):
    coef_df = pd.DataFrame({"Feature": model_columns, "Coefficient": model.coef_})
    coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
    st.bar_chart(coef_df.set_index("Feature"))

# --- Scatter Plot ---
with st.expander("Scatter Plot: Actual vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="#007bff", alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Actual vs Predicted Salary")
    st.pyplot(fig)

# --- Line Plot ---
with st.expander("Line Plot: Prediction Over Samples"):
    fig, ax = plt.subplots()
    ax.plot(y_test.reset_index(drop=True), label="Actual", color='blue', marker='o')
    ax.plot(pd.Series(y_pred), label="Predicted", color='red', marker='x')
    ax.set_title("Actual vs Predicted Salary Over Samples")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)
