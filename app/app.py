import streamlit as st
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

st.set_page_config(page_title="Student Digital Twin", layout="centered")
st.title("Student Digital Twin: Exam Success Prediction")
st.markdown("A simplified digital twin using behavioral and academic features to predict exam success probability.")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
cols = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
means = joblib.load(os.path.join(MODEL_DIR, "feature_means.pkl"))

st.sidebar.header("Student Profile (Digital Twin)")
hours = st.sidebar.slider("Hours Studied per Week", 0, 50, 10)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
sleep = st.sidebar.slider("Sleep Hours per Night", 0, 12, 7)
prev = st.sidebar.slider("Previous Exam Score", 0, 100, 70)
tutor = st.sidebar.slider("Tutoring Sessions per Month", 0, 20, 2)
activity = st.sidebar.slider("Physical Activity (h/week)", 0, 20, 3)

st.header("Prediction Result")

if st.button("Predict Exam Success", type="primary", use_container_width=True):
    x = means.copy().reshape(1, -1)
    
    feature_map = {
        'Hours_Studied': hours, 'Attendance': attendance,
        'Sleep_Hours': sleep, 'Previous_Scores': prev,
        'Tutoring_Sessions': tutor, 'Physical_Activity': activity
    }
    for i, col in enumerate(cols):
        if col in feature_map:
            x[0, i] = feature_map[col]

    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Passing", f"{prob:.1%}")
        st.progress(float(prob))
    with col2:
        if prob >= 0.7:
            st.success("Risk: LOW")
        elif prob >= 0.4:
            st.warning("Risk: MEDIUM")
        else:
            st.error("Risk: HIGH")

    st.subheader("Student Profile Summary")
    st.write(f"Hours Studied: {hours} h/week")
    st.write(f"Attendance: {attendance}%")
    st.write(f"Sleep: {sleep} h/night")
    st.write(f"Previous Scores: {prev}%")
    st.write(f"Tutoring: {tutor} sessions/month")
    st.write(f"Physical Activity: {activity} h/week")

st.markdown("---")
st.caption("Student Digital Twin Project | CSS 324: Introduction to Machine Learning")
