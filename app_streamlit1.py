import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_resources()

st.set_page_config(page_title="EEG Lie Detector", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 EEG Lie Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #306998;'>Predict if a person is lying based on EEG signals</p>", unsafe_allow_html=True)
st.write("---")

# upload a csv file

uploaded_file = st.file_uploader("Upload CSV file with EEG data (columns: EEG.AF3, EEG.T7, EEG.Pz, EEG.T8, EEG.AF4)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")
    
    expected_cols = ["EEG.AF3","EEG.T7","EEG.Pz","EEG.T8","EEG.AF4"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing expected columns: {missing_cols}")
    else:
        st.success("All expected EEG columns found!")

        if st.button("Predict CSV"):
            features = df[expected_cols].values.astype(float)
            
            if scaler is not None:
                features = scaler.transform(features)
            
            predictions = model.predict(features)
            
            if hasattr(model, "predict_proba"):
                confidences = model.predict_proba(features).max(axis=1)
            else:
                confidences = [None]*len(predictions)
            
            df["Prediction"] = ["🤥 Lie" if p==1 else "✅ Truth" for p in predictions]
            df["Confidence"] = confidences
            
            st.dataframe(df)

else:
    st.write("Upload a CSV file to start prediction.")

st.write("---")

# button to get the data from the device
if st.button("📡 Get Live EEG from Crown"):
    eeg_data = get_live_eeg()
    if eeg_data:
        features = np.array([eeg_data])
        if scaler is not None:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]
        label = "🤥 Lie" if prediction == 1 else "✅ Truth"

        st.write("Live EEG Data:", eeg_data)
        st.success(f"Prediction: {label}")
    else:
        st.error("Failed to fetch EEG data from Crown API")


st.markdown("### Or input EEG signals manually")

col1, col2, col3 = st.columns(3)

with col1:
    eeg_af3 = st.number_input("EEG.AF3", value=0.0, format="%.5f", step=0.001)
    eeg_pz = st.number_input("EEG.Pz", value=0.0, format="%.5f", step=0.001)

with col2:
    eeg_t7 = st.number_input("EEG.T7", value=0.0, format="%.5f", step=0.001)
    eeg_t8 = st.number_input("EEG.T8", value=0.0, format="%.5f", step=0.001)

with col3:
    eeg_af4 = st.number_input("EEG.AF4", value=0.0, format="%.5f", step=0.001)
    st.write("")  # empty for alignment

if st.button("🔍 Predict single"):
    features = np.array([[eeg_af3, eeg_t7, eeg_pz, eeg_t8, eeg_af4]])
    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = max(proba)

    label = "🤥 Lie" if prediction == 1 else "✅ Truth"

    if prediction == 1:
        st.markdown(f"<h2 style='color: #D00000;'>Prediction: {label}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color: #008000;'>Prediction: {label}</h2>", unsafe_allow_html=True)

    if confidence is not None:
        st.markdown(f"<h4 style='color: #555555;'>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)
    else:
        st.write("Confidence score not available for this model.")

st.write("---")
st.markdown("<p style='font-size:12px; color:gray; text-align:center;'>Sofija Sekulovska 211100</p>", unsafe_allow_html=True)
