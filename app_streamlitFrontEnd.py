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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EEG Lie Detector",
    page_icon="🧠",
    layout="centered"
)

# --- CUSTOM CSS ---


# --- Custom CSS ---
st.markdown("""
    <style>
        /* Позадина со градиент */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa, #ffffff, #fce4ec);
            font-family: 'Segoe UI', sans-serif;
        }
        

        /* Header стил */
        .main-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
        }
        .main-header img {
            width: 70px;
        }
            
        .main-header h1 {
            font-size: 36px;
            font-weight: 700;
            color: #37474f;
        }

        /* Име стил */
        .author {
            font-size: 22px;
            font-weight: 600;
            color: #00796b;
            text-align: center;
            margin-bottom: 40px;
            font-family: 'Trebuchet MS', sans-serif;
        }

        /* Card стил */
        .content-box {
            background: white;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER со лого и наслов ---
col1, col2, col3 = st.columns([1,3,1])
with col1:
    st.image("finki_logo.png", width=100)  # стави го логото во папка
with col2:
    st.markdown("<h1 style='text-align:center;'>🧠 EEG Lie Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555;'>Predict if a person is lying based on EEG signals</p>", unsafe_allow_html=True)
with col3:
    st.write("")

st.write("---")
# --- AUTHOR ---
#st.markdown("<div class='author'>Изработила: <b>Софија Секуловска</b></div>", unsafe_allow_html=True)

# --- CONTENT (пример секции) ---
st.markdown("<div class='content-box'>📌 Опис на проектот: Ова е систем за детекција на лаги базиран на EEG сигнали.</div>", unsafe_allow_html=True)
st.markdown("<div class='content-box'>⚙️ Методологија: Се користат машински модели за класификација (SVM, Neural Networks, Random Forest).</div>", unsafe_allow_html=True)
st.markdown("<div class='content-box'>📊 Резултати: Моделите се тестирани со Accuracy, Precision, Recall и F1-score.</div>", unsafe_allow_html=True)



# --- HEADER WITH LOGO ---
#col1, col2, col3 = st.columns([1,3,1])
#with col1:
#    st.image("finki_logo.png", width=100)  # стави го логото во папка
#with col2:
#    st.markdown("<h1 style='text-align:center;'>🧠 EEG Lie Detector</h1>", unsafe_allow_html=True)
#    st.markdown("<p style='text-align:center; color:#555;'>Predict if a person is lying based on EEG signals</p>", unsafe_allow_html=True)
#with col3:
#    st.write("")
#
st.write("---")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader(
    "📂 Upload CSV file with EEG data",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

    expected_cols = ["EEG.AF3","EEG.T7","EEG.Pz","EEG.T8","EEG.AF4"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        st.error(f"⚠ Missing expected columns: {missing_cols}")
    else:
        st.success("✅ All expected EEG columns found!")

        if st.button("🔮 Predict CSV"):
            features = df[expected_cols].values.astype(float)
            if scaler is not None:
                features = scaler.transform(features)

            predictions = model.predict(features)
            confidences = model.predict_proba(features).max(axis=1) if hasattr(model, "predict_proba") else [None]*len(predictions)

            df["Prediction"] = ["🤥 Lie" if p==1 else "✅ Truth" for p in predictions]
            df["Confidence"] = confidences

            st.dataframe(df)

else:
    st.info("👉 Please upload a CSV file to start prediction.")

st.write("---")

# --- LIVE EEG BUTTON ---
if st.button("📡 Get Live EEG from Crown"):
    try:
        eeg_data = get_live_eeg()
    except:
        eeg_data = None

    if eeg_data:
        features = np.array([eeg_data])
        if scaler is not None:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]
        label = "🤥 Lie" if prediction == 1 else "✅ Truth"

        st.write("Live EEG Data:", eeg_data)
        st.success(f"Prediction: {label}")
    else:
        st.error("⚠ Failed to fetch EEG data from Crown API")

# --- MANUAL INPUT ---
st.markdown("### ✍ Input EEG signals manually")

col1, col2, col3 = st.columns(3)
with col1:
    eeg_af3 = st.number_input("EEG.AF3", value=0.0, format="%.5f", step=0.001)
    eeg_pz = st.number_input("EEG.Pz", value=0.0, format="%.5f", step=0.001)

with col2:
    eeg_t7 = st.number_input("EEG.T7", value=0.0, format="%.5f", step=0.001)
    eeg_t8 = st.number_input("EEG.T8", value=0.0, format="%.5f", step=0.001)

with col3:
    eeg_af4 = st.number_input("EEG.AF4", value=0.0, format="%.5f", step=0.001)

if st.button("🔍 Predict Single"):
    features = np.array([[eeg_af3, eeg_t7, eeg_pz, eeg_t8, eeg_af4]])
    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = max(proba)

    label = "🤥 Lie" if prediction == 1 else "✅ Truth"
    color = "#D00000" if prediction == 1 else "#008000"

    st.markdown(f"<h2 style='color:{color};'>Prediction: {label}</h2>", unsafe_allow_html=True)
    if confidence is not None:
        st.markdown(f"<h4 style='color:#555;'>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(
    "<p style='font-size:12px; color:gray; text-align:center; bottom: 50px; font-style:italic; font-family:Georgia, serif;'>Sofija Sekulovska 211100</p>", 
    unsafe_allow_html=True
)



