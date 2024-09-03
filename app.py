import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
classifier = joblib.load('dbn_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you saved the scaler
metrics = joblib.load('metrics.pkl')

# Streamlit UI
st.title("Heart Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.number_input("Resting ECG Results (restecg)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=300, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0)
slope = st.number_input("Slope of the peak exercise ST segment (slope)", min_value=0, max_value=2, value=0)
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3, value=0)

# Collect input data and scale it
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
input_data = scaler.transform([input_data])  # Apply scaling

# Function for predicting heart disease
def predict_heart_disease(model, input_data):
    prediction = model.predict(input_data)
    return "Yes" if prediction[0] == 1 else "No"

# Predict button
if st.button("Predict"):
    result = predict_heart_disease(classifier, input_data)
    st.write(f"Heart Disease Prediction: **{result}**")

# Display metrics
st.subheader("Model Metrics")
st.write(f"Accuracy: {metrics['accuracy']:.2f}")
st.write(f"Loss (Log Loss): {metrics['loss']:.2f}")
st.write(f"Precision: {metrics['precision']:.2f}")
st.write(f"F1 Score: {metrics['f1_score']:.2f}")
