import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction System")

# Load model and preprocessing tools
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le_sex = joblib.load("model/le_sex.pkl")
le_embarked = joblib.load("model/le_embarked.pkl")

st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=7.25)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]

    features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
