# Wine Cultivar Origin Prediction System
# Streamlit Web Application

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Wine Cultivar Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #722F37;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model, scaler, and feature names
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('model/wine_cultivar_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        feature_names = joblib.load('model/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, feature_names = load_model_components()

# Header
st.title("🍷 Wine Cultivar Origin Prediction System")
st.markdown("### Predict wine cultivar based on chemical properties")
st.markdown("---")

# Check if model is loaded
if model is None or scaler is None or feature_names is None:
    st.error("⚠️ Failed to load model components. Please ensure all model files are in the 'model/' directory.")
    st.stop()

# Feature descriptions
feature_info = {
    'alcohol': {'min': 11.0, 'max': 15.0, 'default': 13.0, 'unit': '%'},
    'malic_acid': {'min': 0.5, 'max': 6.0, 'default': 2.5, 'unit': 'g/L'},
    'flavanoids': {'min': 0.3, 'max': 5.5, 'default': 2.0, 'unit': 'g/L'},
    'color_intensity': {'min': 1.0, 'max': 13.0, 'default': 5.0, 'unit': ''},
    'hue': {'min': 0.4, 'max': 1.7, 'default': 1.0, 'unit': ''},
    'proline': {'min': 250.0, 'max': 1700.0, 'default': 750.0, 'unit': 'mg/L'}
}

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system predicts the cultivar (origin) of wine samples based on their chemical properties.
    
    **Model Information:**
    - Algorithm: Random Forest Classifier
    - Features: 6 chemical properties
    - Classes: 3 cultivars (0, 1, 2)
    """)
    
    st.markdown("---")
    st.header("📊 Feature Ranges")
    for feature, info in feature_info.items():
        st.write(f"**{feature.replace('_', ' ').title()}**")
        st.write(f"Range: {info['min']} - {info['max']} {info['unit']}")
        st.write("")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Wine Chemical Properties")
    
    # Create input fields
    input_data = {}
    
    # Organize inputs in 2 columns
    input_col1, input_col2 = st.columns(2)
    
    features_list = list(feature_info.keys())
    
    for idx, feature in enumerate(features_list):
        info = feature_info[feature]
        col = input_col1 if idx % 2 == 0 else input_col2
        
        with col:
            input_data[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()} ({info['unit']})",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(info['default']),
                step=0.1,
                key=feature
            )
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Cultivar", use_container_width=True)
    
    # Prediction logic
    if predict_button:
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct feature order
            input_df = input_df[feature_names]
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Main prediction
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #722F37; color: white; text-align: center;'>
                <h2>Predicted Cultivar: {prediction}</h2>
                <p style='font-size: 18px;'>Confidence: {prediction_proba[prediction]*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability distribution
            st.write("**Probability Distribution:**")
            prob_df = pd.DataFrame({
                'Cultivar': [f'Cultivar {i}' for i in range(3)],
                'Probability': prediction_proba * 100
            })
            
            st.bar_chart(prob_df.set_index('Cultivar'))
            
            # Detailed probabilities
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            with col_prob1:
                st.metric("Cultivar 0", f"{prediction_proba[0]*100:.2f}%")
            with col_prob2:
                st.metric("Cultivar 1", f"{prediction_proba[1]*100:.2f}%")
            with col_prob3:
                st.metric("Cultivar 2", f"{prediction_proba[2]*100:.2f}%")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

with col2:
    st.subheader("📋 Sample Values")
    st.info("""
    **Cultivar 0 Sample:**
    - Alcohol: 14.2%
    - Malic Acid: 1.7 g/L
    - Flavanoids: 3.0 g/L
    - Color Intensity: 4.5
    - Hue: 1.0
    - Proline: 1065 mg/L
    
    **Cultivar 1 Sample:**
    - Alcohol: 12.5%
    - Malic Acid: 2.5 g/L
    - Flavanoids: 1.8 g/L
    - Color Intensity: 5.5
    - Hue: 0.8
    - Proline: 625 mg/L
    
    **Cultivar 2 Sample:**
    - Alcohol: 13.2%
    - Malic Acid: 3.2 g/L
    - Flavanoids: 1.2 g/L
    - Color Intensity: 7.5
    - Hue: 0.6
    - Proline: 520 mg/L
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Wine Cultivar Prediction System | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
