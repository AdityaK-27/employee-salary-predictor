
import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load("best_model.pkl")

# Set title
st.title("💼 Employee Salary Prediction App")
st.write("Enter the employee's details below to predict if their salary is **≥50K or <50K**.")

# Input form
with st.form("salary_form"):
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                           'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                           'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                                           '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                     'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                             'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                             'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                             'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    hours_per_week = st.slider("Hours per week", min_value=1, max_value=100, value=40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany',
                                                     'Canada', 'India', 'England', 'China', 'Other'])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)

    submit = st.form_submit_button("Predict Salary")

# Prediction
if submit:
    # Create input DataFrame
    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display result
    if prediction == 1:
        st.success(f"✅ The model predicts this person is likely to earn **≥50K**. (Confidence: {prob:.2f})")
    else:
        st.warning(f"❌ The model predicts this person is likely to earn **<50K**. (Confidence: {1 - prob:.2f})")
