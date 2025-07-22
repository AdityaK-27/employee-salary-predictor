import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Configure page settings
st.set_page_config(
    page_title="Employee Salary Estimator",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load trained model with error handling
@st.cache_resource
def load_model():
    """Load the trained LightGBM model with caching for better performance."""
    try:
        model = joblib.load("model_lightgbm.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'model_lightgbm.pkl' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Professional CSS styling
def inject_custom_css():
    """Inject custom CSS for professional appearance."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            font-weight: 300;
            opacity: 0.9;
        }
        
        /* Form container */
        .form-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
            border: 1px solid #e9ecef;
        }
        
        /* Section headers */
        .section-header {
            color: #2c3e50;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            height: 3.5rem;
            width: 100%;
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Success message styling */
        .salary-result {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
        }
        
        /* Input styling */
        .stSelectbox > div > div > div {
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        .stSlider > div > div > div > div {
            background-color: #667eea;
        }
        
        .stNumberInput > div > div > input {
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        /* Chart container */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin-top: 1rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid #e9ecef;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .header-title {
                font-size: 2rem;
            }
            .form-container {
                padding: 1.5rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

# Data constants
class DataConstants:
    """Constants for dropdown options to maintain data integrity."""
    
    WORKCLASS_OPTIONS = [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ]
    
    EDUCATION_OPTIONS = [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", 
        "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", 
        "5th-6th", "10th", "1st-4th", "Preschool", "12th"
    ]
    
    MARITAL_STATUS_OPTIONS = [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", 
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ]
    
    OCCUPATION_OPTIONS = [
        "Tech-support", "Craft-repair", "Other-service", "Sales", 
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
        "Transport-moving", "Priv-house-serv", "Protective-serv", 
        "Armed-Forces", "Unknown"
    ]
    
    RELATIONSHIP_OPTIONS = [
        "Wife", "Own-child", "Husband", "Not-in-family", 
        "Other-relative", "Unmarried"
    ]
    
    RACE_OPTIONS = [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ]
    
    NATIVE_COUNTRY_OPTIONS = [
        "United-States", "Mexico", "Philippines", "Germany", "Canada", 
        "India", "China", "Cuba", "England", "Japan", "South"
    ]

def render_header():
    """Render the professional header section."""
    st.markdown("""
        <div class="header-container">
            <div class="header-title">💼 Employee Salary Estimator</div>
            <div class="header-subtitle">
                Advanced ML-powered salary prediction system for accurate compensation estimates
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_form_section(title: str, content_func):
    """Render a form section with consistent styling."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    content_func()

def collect_user_inputs() -> Tuple[np.ndarray, dict]:
    """Collect and validate user inputs with organized sections."""
    
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Personal Information Section
    def personal_info():
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 90, 30, help="Employee's current age")
            sex = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            race = st.selectbox("Race", DataConstants.RACE_OPTIONS)
            native_country = st.selectbox("Native Country", DataConstants.NATIVE_COUNTRY_OPTIONS)
        return age, sex, race, native_country
    
    # Professional Information Section  
    def professional_info():
        col1, col2 = st.columns(2)
        with col1:
            workclass = st.selectbox("Work Class", DataConstants.WORKCLASS_OPTIONS, 
                                   help="Type of employment")
            occupation = st.selectbox("Occupation", DataConstants.OCCUPATION_OPTIONS)
            hours_per_week = st.slider("Hours Per Week", 1, 99, 40, 
                                     help="Average working hours per week")
        with col2:
            education = st.selectbox("Education Level", DataConstants.EDUCATION_OPTIONS)
            education_num = st.slider("Education Number", 1, 16, 9, 
                                    help="Numerical representation of education level")
            fnlwgt = st.number_input("Final Weight (fnlwgt)", 0, 
                                   help="Demographic weighting factor")
        return workclass, occupation, hours_per_week, education, education_num, fnlwgt
    
    # Personal Status Section
    def personal_status():
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox("Marital Status", DataConstants.MARITAL_STATUS_OPTIONS)
            relationship = st.selectbox("Relationship Status", DataConstants.RELATIONSHIP_OPTIONS)
        with col2:
            capital_gain = st.number_input("Capital Gain", 0, help="Annual capital gains")
            capital_loss = st.number_input("Capital Loss", 0, help="Annual capital losses")
        return marital_status, relationship, capital_gain, capital_loss
    
    # Collect inputs by sections
    render_form_section("👤 Personal Information", 
                       lambda: personal_info())
    age, sex, race, native_country = personal_info()
    
    render_form_section("💼 Professional Information", 
                       lambda: professional_info())
    workclass, occupation, hours_per_week, education, education_num, fnlwgt = professional_info()
    
    render_form_section("📊 Personal Status", 
                       lambda: personal_status())
    marital_status, relationship, capital_gain, capital_loss = personal_status()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Encode categorical variables
    encoded_inputs = {
        'age': age,
        'workclass': DataConstants.WORKCLASS_OPTIONS.index(workclass),
        'education': DataConstants.EDUCATION_OPTIONS.index(education),
        'education_num': education_num,
        'marital_status': DataConstants.MARITAL_STATUS_OPTIONS.index(marital_status),
        'occupation': DataConstants.OCCUPATION_OPTIONS.index(occupation),
        'relationship': DataConstants.RELATIONSHIP_OPTIONS.index(relationship),
        'race': DataConstants.RACE_OPTIONS.index(race),
        'sex': 1 if sex == "Male" else 0,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': DataConstants.NATIVE_COUNTRY_OPTIONS.index(native_country),
        'fnlwgt': fnlwgt
    }
    
    # Create input array in correct order for model
    input_array = np.array([[
        encoded_inputs['age'], encoded_inputs['workclass'], encoded_inputs['education'],
        encoded_inputs['education_num'], encoded_inputs['marital_status'], 
        encoded_inputs['occupation'], encoded_inputs['relationship'], 
        encoded_inputs['race'], encoded_inputs['sex'], encoded_inputs['capital_gain'],
        encoded_inputs['capital_loss'], encoded_inputs['hours_per_week'], 
        encoded_inputs['native_country'], encoded_inputs['fnlwgt']
    ]])
    
    return input_array, encoded_inputs

def create_professional_chart(salary_value: int, prediction_class: str):
    """Create a professional-looking salary chart."""
    
    # Set the style for better aesthetics
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create gradient bar
    colors = ['#667eea', '#764ba2']
    bars = ax.bar(['Predicted Monthly Salary'], [salary_value], 
                  color=colors[0], alpha=0.8, width=0.6)
    
    # Customize the chart
    ax.set_ylabel('Amount (INR)', fontsize=12, fontweight='bold')
    ax.set_title(f'Salary Prediction Result - {prediction_class}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100000)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'₹{height:,.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Customize grid and spines
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Format y-axis to show currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    plt.tight_layout()
    return fig

def predict_salary(input_data: np.ndarray) -> Tuple[str, int, str]:
    """Make salary prediction and return formatted results."""
    try:
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            salary_display = "₹75,000/month"
            salary_value = 75000
            prediction_class = "High Income Group"
        else:
            salary_display = "₹30,000/month"
            salary_value = 30000
            prediction_class = "Lower Income Group"
        
        return salary_display, salary_value, prediction_class
    
    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
        return "Error", 0, "Unknown"

def render_footer():
    """Render professional footer."""
    st.markdown("""
        <div class="footer">
            <p>🔬 Powered by Advanced Machine Learning | Built with Streamlit</p>
            <p><small>This tool provides estimates based on statistical models and should be used as a reference only.</small></p>
        </div>
    """, unsafe_allow_html=True)

def main():
    """Main application logic."""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render header
    render_header()
    
    # Collect user inputs
    input_data, input_details = collect_user_inputs()
    
    # Prediction button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Estimate Salary", key="predict_btn"):
            with st.spinner("Analyzing data and generating prediction..."):
                salary_display, salary_value, prediction_class = predict_salary(input_data)
                
                if salary_value > 0:
                    # Display result
                    st.markdown(f"""
                        <div class="salary-result">
                            🎯 Estimated Salary: {salary_display}<br>
                            <small>Category: {prediction_class}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create and display professional chart
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("📊 Salary Analysis Visualization")
                    
                    chart = create_professional_chart(salary_value, prediction_class)
                    st.pyplot(chart, use_container_width=True)
                    plt.close()  # Clean up to prevent memory issues
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional insights
                    with st.expander("📈 View Prediction Insights"):
                        st.write("**Model Confidence:** Based on historical employment data")
                        st.write("**Prediction Factors:** Age, Education, Occupation, Work Hours, and other demographic factors")
                        st.write("**Note:** This is an estimate based on statistical patterns and should be used as a reference.")
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
