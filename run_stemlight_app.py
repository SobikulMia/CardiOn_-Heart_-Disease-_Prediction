
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and artifacts
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
except:
    st.error("Model artifacts not found. Please run model_training.py first")
    st.stop()

# Initialize session state for patient data
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []

# Helper functions

def predict_risk(input_data):
    processed_data = preprocess_input(input_data)
    scaled_data = scaler.transform([processed_data])
    proba = model.predict_proba(scaled_data)[0][1]
    return proba


def preprocess_input(input_data):
    processed = {}
    processed['age_year'] = input_data['age']
    processed['gender'] = 1 if input_data['gender'] == 'Male' else 2
    processed['height'] = input_data['height']
    processed['weight'] = input_data['weight']
    systolic, diastolic = map(int, input_data['bp'].split('/'))
    processed['ap_hi'] = systolic
    processed['ap_lo'] = diastolic

    chol_map = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    processed['cholesterol'] = chol_map[input_data['cholesterol']]

    gluc_map = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    processed['gluc'] = gluc_map[input_data['gluc']]

    processed['smoke'] = int(input_data['smoke'])
    processed['alco'] = int(input_data['alco'])
    processed['active'] = int(input_data['active'])

    height_m = input_data['height'] / 100
    bmi = input_data['weight'] / (height_m ** 2)
    processed['BMI'] = bmi
    processed['is_obese'] = int(bmi > 30)

    processed['pulse_pressure'] = systolic - diastolic
    processed['map'] = (2 * diastolic + systolic) / 3
    processed['bp_ratio'] = systolic / diastolic if diastolic != 0 else 0
    processed['bp_diff_ratio'] = (systolic - diastolic) / systolic if systolic != 0 else 0

    final_input = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        final_input[col] = [processed.get(col, 0)]

    return final_input.values[0]

st.set_page_config(layout="wide", page_title="CardiOn: Heart Disease Prediction", page_icon="❤️")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Patient Prediction", "Model Analysis"])

if page == "Dashboard":
    st.title("❤️ CardiOn: Heart Disease Prediction Dashboard")

    total_patients = len(st.session_state['predictions'])
    high_risk = sum(1 for p in st.session_state['predictions'] if p['Prediction'] == 'High Risk')
    medium_risk = sum(1 for p in st.session_state['predictions'] if p['Prediction'] == 'Medium Risk')
    low_risk = total_patients - high_risk - medium_risk

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_patients)
    col2.metric("High Risk Patients", high_risk)
    col3.metric("Medium Risk Patients", medium_risk)

    risk_data = pd.DataFrame({
        'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
        'Patients': [low_risk, medium_risk, high_risk]
    })

    fig = px.pie(risk_data, values='Patients', names='Risk Level', color_discrete_sequence=['#00b894', '#fdcb6e', '#d63031'])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame(st.session_state['predictions']))

elif page == "Patient Prediction":
    st.title("🫸 Patient Risk Prediction")

    age = st.slider("Age", 20, 100, 45)
    gender = st.radio("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 140, 220, 170)
    weight = st.number_input("Weight (kg)", 40, 150, 70)
    bp = st.text_input("Blood Pressure (systolic/diastolic)", "120/80")
    cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.checkbox("Smoker")
    alco = st.checkbox("Alcohol Consumer")
    active = st.checkbox("Physically Active")

    if st.button("Calculate Risk"):
        input_data = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'bp': bp,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }

        risk_score = predict_risk(input_data)

        if risk_score < 0.3:
            prediction = 'Low Risk'
        elif risk_score < 0.6:
            prediction = 'Medium Risk'
        else:
            prediction = 'High Risk'

        st.session_state['predictions'].append({
            'Age': age,
            'Gender': gender,
            'BP': bp,
            'Cholesterol': cholesterol,
            'Glucose': gluc,
            'Risk Score': round(risk_score, 2),
            'Prediction': prediction
        })

        st.success(f"Risk Level: {prediction} (Score: {risk_score:.2f})")

elif page == "Model Analysis":
    st.title(" Model Performance Analysis")
    metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.73, 0.73, 0.73, 0.73]
        
    })
    st.dataframe(metrics)

    fig = px.bar(metrics, x='Metric', y='Value', color='Value', color_continuous_scale='Bluered', text='Value', range_y=[0,1])
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    features = pd.DataFrame({
        'Feature': ['Systolic BP', 'Age', 'Cholesterol', 'BMI', 'Smoking', 'Physical Inactivity'],
        'Importance': [0.32, 0.25, 0.18, 0.12, 0.08, 0.05]
    })

    fig2 = px.treemap(features, path=['Feature'], values='Importance', color='Importance', color_continuous_scale='RdBu')
    st.plotly_chart(fig2, use_container_width=True)
