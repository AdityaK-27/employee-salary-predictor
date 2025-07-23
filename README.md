# 💼 Salary Prediction Web App

A simple and professional web application built with **Streamlit** that predicts employee salaries based on demographic and job-related inputs. The app uses a **Linear Regression** model trained on real-world data and provides interactive visualizations along with key performance metrics.

---

## Live Demo

You can try the app here: [Streamlit App](https://employee-salary-predictor-adityak-27.streamlit.app/)

---
## 📌 Features

- Predicts salary based on:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
- Displays key model metrics (MAE, MSE, R²)
- Responsive sidebar with detailed information
- Clean, form-based input interface for ease of use
- Saves and visualizes model predictions

---

## 🧠 Model Details

- **Algorithm:** Linear Regression
- **Training Data:** Cleaned CSV file with categorical encoding
- **Performance:**
  - **MAE:** 11,596.52
  - **MSE:** 354,248,539.01
  - **R² Score:** 0.8522
- Trained using scikit-learn and exported via `joblib`.

---

## 📁 Project Structure
```
├── Salary Data.csv # Original dataset
├── app.py # Streamlit application
├── salary_prediction_model.pkl # Trained regression model
├── model_columns.pkl # Feature columns used by the model
├── test_predictions.pkl # Saved test set predictions for plotting
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```


---

## 🚀 Getting Started

### 1. Clone the repository
```
git clone https://github.com/AdityaK-27/employee-salary-predictor/tree/main.git
cd salary-prediction-app
pip install -r requirements.txt
streamlit run app.py
```
## 🖥️ Screenshot

<img width="1919" height="1071" alt="image" src="https://github.com/user-attachments/assets/c7949bcc-8687-44a6-a542-be50ea622590" />

---

## 📊 Sample Prediction Inputs

| Field            | Example        |
|------------------|----------------|
| Age              | 32             |
| Gender           | Male           |
| Education Level  | Master's       |
| Job Title        | Data Scientist |
| Experience       | 6              |

---

## 🛠️ Tools & Libraries

- **Streamlit** – Web UI  
- **Scikit-learn** – Model training  
- **Pandas** – Data handling  
- **Joblib** – Model serialization  
- **Matplotlib** – Plotting test results  

---

## 📬 Contact

**Developer:** Aditya Kankarwal  
**Email:** aditya27kankarwal@gmail.com  
**LinkedIn:** [linkedin.com](https://www.linkedin.com/in/aditya-kankarwal-68b626300/)
