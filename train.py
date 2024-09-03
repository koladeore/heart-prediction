import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

# Load the dataset
df = pd.read_csv('heart.csv')

# Split data into features and target variable
X = df.drop(columns='target')
y = df['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the DBN model using BernoulliRBM
rbm = BernoulliRBM(random_state=42)
logistic = LogisticRegression(max_iter=1000, random_state=42)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
loss = log_loss(y_test, classifier.predict_proba(X_test))
precision = class_report['1']['precision']
f1_score = class_report['1']['f1-score']

# Save metrics and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(classifier, 'dbn_model.pkl')
joblib.dump({'accuracy': accuracy, 'loss': loss, 'precision': precision, 'f1_score': f1_score}, 'metrics.pkl')

# Streamlit UI
st.title("Heart Disease Prediction")

# User input fields
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
metrics = joblib.load('metrics.pkl')
st.write(f"Accuracy: {metrics['accuracy']:.2f}")
st.write(f"Loss (Log Loss): {metrics['loss']:.2f}")
st.write(f"Precision: {metrics['precision']:.2f}")
st.write(f"F1 Score: {metrics['f1_score']:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.write(class_report)
