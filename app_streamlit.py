# # app_streamlit.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import time
# import joblib
# import os
# from collections import deque

# # If using Keras:
# import tensorflow as tf

# MODEL_PATH = "models/model.pkl"
# SCALER_PATH = "models/scaler.pkl"  # optional

# @st.cache(allow_output_mutation=True)
# def load_resources():
#     model = tf.keras.models.load_model(MODEL_PATH)
#     scaler = None
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#     return model, scaler

# model, scaler = load_resources()

# st.title("EEG Lie Detector — Real-time Simulation")
# st.write("This demo streams rows from a CSV and shows live predictions.")

# uploaded = st.file_uploader("Upload mock CSV (or leave empty to use data/mock_data.csv)", type="csv")
# csv_path = uploaded if uploaded else "mock_data.csv"
# window_size = st.number_input("Window size (timesteps)", min_value=1, value=50)
# delay = st.slider("Delay between rows (seconds)", min_value=0.01, max_value=2.0, value=0.2)
# start = st.button("Start simulation")

# # def get_channel_cols(df):
# #     # heuristic: treat columns that start with ch or eeg as channels
# #     cols = [c for c in df.columns if c.lower().startswith("ch") or c.lower().startswith("eeg")]
# #     if not cols:
# #         # fallback: all numeric except label
# #         cols = [c for c in df.columns if c != "label"]
# #     return cols
# def get_channel_cols(df):
#     cols = [c for c in df.columns if c.lower().startswith("ch") or c.lower().startswith("eeg") or c.lower().startswith("eeg.")]
#     if not cols:
#         # fallback: all numeric columns except 'label'
#         cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
#     return cols

# df = pd.read_csv(csv_path)
# st.write("CSV loaded with shape:", df.shape)
# channel_cols = get_channel_cols(df)
# st.write("Detected EEG channel columns:", channel_cols)


# if start:
#     df = pd.read_csv(csv_path)
#     channel_cols = get_channel_cols(df)
#     st.write(f"Using channel columns: {channel_cols}")

#     buffer = deque(maxlen=window_size)
#     placeholder_chart = st.empty()
#     placeholder_label = st.empty()
#     placeholder_conf = st.empty()
#     logs = []

#     for idx, row in df.iterrows():
#         sample = row[channel_cols].astype(float).values
#         buffer.append(sample)

#         # draw chart for current window
#         win_df = pd.DataFrame(list(buffer), columns=channel_cols)
#         placeholder_chart.line_chart(win_df)

#         # only predict when buffer is full (for sequence models)
#         if len(buffer) == window_size:
#             X = np.array(win_df)  # shape (window_size, n_channels)
#             # prepare input depending on model expected dims
#             input_shape = model.input_shape  # e.g. (None, timesteps, channels) or (None, features)
#             if len(input_shape) == 3:
#                 # model expects (batch, timesteps, channels)
#                 if scaler is not None:
#                     flat = X.reshape(-1, X.shape[1])
#                     scaled = scaler.transform(flat).reshape(1, X.shape[0], X.shape[1])
#                     X_input = scaled
#                 else:
#                     X_input = X.reshape(1, X.shape[0], X.shape[1])
#             else:
#                 # model expects 2D features (batch, features) — flatten the window
#                 flat = X.flatten().reshape(1, -1)
#                 if scaler is not None:
#                     X_input = scaler.transform(flat)
#                 else:
#                     X_input = flat

#             pred = model.predict(X_input)
#             # handle binary vs multi-class outputs
#             if pred.shape[-1] == 1:
#                 prob = float(pred[0][0])
#                 label = "Lie" if prob > 0.5 else "Truth"
#                 conf = prob if prob > 0.5 else 1 - prob
#             else:
#                 idx_max = int(np.argmax(pred, axis=1)[0])
#                 conf = float(pred[0][idx_max])
#                 # assume class 1 == Lie, class 0 == Truth — change as needed
#                 label = "Lie" if idx_max == 1 else "Truth"

#             placeholder_label.metric("Prediction", label)
#             placeholder_conf.write(f"Confidence: {conf:.2f}")

#             logs.append({"idx": int(idx), "label": label, "conf": float(conf)})

#         time.sleep(delay)

#     st.write("Simulation finished. Prediction log:")
#     st.dataframe(pd.DataFrame(logs))


# ..........................................
# import streamlit as st
# import pandas as pd
# import numpy as np
# from collections import deque
# import os
# import joblib
# import tensorflow as tf

# MODEL_PATH = "models/model.pkl"
# SCALER_PATH = "models/scaler.pkl"  # optional

# @st.cache_resource
# def load_resources():
#     import joblib
#     model = joblib.load(MODEL_PATH)  # load sklearn model
#     scaler = None
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#     return model, scaler


# model, scaler = load_resources()

# st.title("EEG Lie Detector — Real-time Simulation")

# uploaded = st.file_uploader("Upload mock CSV (or leave empty to use data/mock_data.csv)", type="csv")
# csv_path = uploaded if uploaded else "mock_data.csv"
# window_size = st.number_input("Window size (timesteps)", min_value=1, value=50)
# delay_ms = st.slider("Delay between rows (milliseconds)", min_value=10, max_value=2000, value=200)
# start = st.button("Start simulation")

# def get_channel_cols(df):
#     cols = [c for c in df.columns if c.lower().startswith("ch") or c.lower().startswith("eeg") or c.lower().startswith("eeg.")]
#     if not cols:
#         cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
#     return cols

# if 'idx' not in st.session_state:
#     st.session_state.idx = 0
#     st.session_state.logs = []
#     st.session_state.buffer = deque(maxlen=window_size)

# if start:
#     df = pd.read_csv(csv_path)
#     channel_cols = get_channel_cols(df)
#     st.write(f"Using channel columns: {channel_cols}")

#     placeholder_chart = st.empty()
#     placeholder_label = st.empty()
#     placeholder_conf = st.empty()

#     if st.session_state.idx < len(df):
#         row = df.iloc[st.session_state.idx]
#         sample = row[channel_cols].astype(float).values
#         st.session_state.buffer.append(sample)

#         win_df = pd.DataFrame(list(st.session_state.buffer), columns=channel_cols)
#         placeholder_chart.line_chart(win_df)

#         if len(st.session_state.buffer) == window_size:
#             X = np.array(win_df)
#             input_shape = model.input_shape
#             if len(input_shape) == 3:
#                 if scaler is not None:
#                     flat = X.reshape(-1, X.shape[1])
#                     scaled = scaler.transform(flat).reshape(1, X.shape[0], X.shape[1])
#                     X_input = scaled
#                 else:
#                     X_input = X.reshape(1, X.shape[0], X.shape[1])
#             else:
#                 flat = X.flatten().reshape(1, -1)
#                 if scaler is not None:
#                     X_input = scaler.transform(flat)
#                 else:
#                     X_input = flat

#             pred = model.predict(X_input)
#             if pred.shape[-1] == 1:
#                 prob = float(pred[0][0])
#                 label = "Lie" if prob > 0.5 else "Truth"
#                 conf = prob if prob > 0.5 else 1 - prob
#             else:
#                 idx_max = int(np.argmax(pred, axis=1)[0])
#                 conf = float(pred[0][idx_max])
#                 label = "Lie" if idx_max == 1 else "Truth"

#             placeholder_label.metric("Prediction", label)
#             placeholder_conf.write(f"Confidence: {conf:.2f}")

#             st.session_state.logs.append({"idx": int(st.session_state.idx), "label": label, "conf": float(conf)})

#         st.session_state.idx += 1

#         # Pause for delay_ms milliseconds
#         st.experimental_rerun()

#     else:
#         st.write("Simulation finished. Prediction log:")
#         st.dataframe(pd.DataFrame(st.session_state.logs))
#         if st.button("Restart simulation"):
#             st.session_state.idx = 0
#             st.session_state.logs = []
#             st.session_state.buffer = deque(maxlen=window_size)
# else:
#     st.write("Press Start simulation to begin streaming EEG data.")

# .............................................................................
# import streamlit as st
# import numpy as np
# import joblib

# MODEL_PATH = "models/model.pkl"

# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)

# model = load_model()

# st.title("EEG Lie Detector")

# # Inputs for EEG channels
# eeg_af3 = st.number_input("EEG.AF3", value=0.0)
# eeg_t7 = st.number_input("EEG.T7", value=0.0)
# eeg_pz = st.number_input("EEG.Pz", value=0.0)
# eeg_t8 = st.number_input("EEG.T8", value=0.0)
# eeg_af4 = st.number_input("EEG.AF4", value=0.0)

# if st.button("Predict"):
#     features = np.array([[eeg_af3, eeg_t7, eeg_pz, eeg_t8, eeg_af4]])
#     prediction = model.predict(features)[0]
#     confidence = max(model.predict_proba(features)[0]) if hasattr(model, "predict_proba") else None

#     label = "Lie" if prediction == 1 else "Truth"
#     st.write(f"Prediction: **{label}**")
#     if confidence is not None:
#         st.write(f"Confidence: {confidence:.2f}")

# .............................................. same as the last but without style
# import streamlit as st
# import numpy as np
# import joblib
# import os

# MODEL_PATH = "models/model.pkl"
# SCALER_PATH = "models/scaler.pkl"

# @st.cache_resource
# def load_resources():
#     model = joblib.load(MODEL_PATH)
#     scaler = None
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#     return model, scaler

# model, scaler = load_resources()

# st.title("EEG Lie Detector")

# # Inputs for the EEG channels (adjust names & defaults as needed)
# eeg_af3 = st.number_input("EEG.AF3", value=0.0, format="%.5f")
# eeg_t7 = st.number_input("EEG.T7", value=0.0, format="%.5f")
# eeg_pz = st.number_input("EEG.Pz", value=0.0, format="%.5f")
# eeg_t8 = st.number_input("EEG.T8", value=0.0, format="%.5f")
# eeg_af4 = st.number_input("EEG.AF4", value=0.0, format="%.5f")

# if st.button("Predict"):
#     features = np.array([[eeg_af3, eeg_t7, eeg_pz, eeg_t8, eeg_af4]])
#     if scaler is not None:
#         features = scaler.transform(features)

#     prediction = model.predict(features)[0]

#     # Try to get confidence from predict_proba if available
#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(features)[0]
#         confidence = max(proba)
#     else:
#         confidence = None

#     label = "Lie" if prediction == 1 else "Truth"

#     st.markdown(f"### Prediction: **{label}**")
#     if confidence is not None:
#         st.markdown(f"### Confidence: **{confidence:.2f}**")
#     else:
#         st.write("Confidence score not available for this model.")

#  With style :)

import streamlit as st
import numpy as np
import joblib
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"



# with data from the crown device - API
import requests
import json

CROWN_API_URL = "https://api.neurosity.co/v1/signal"   
CROWN_API_TOKEN = "CROWN_API_KEY"  

def get_live_eeg():
    headers = {"Authorization": f"Bearer {CROWN_API_TOKEN}"}
    response = requests.get(CROWN_API_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # kolonite
        return [data.get("EEG.AF3"), data.get("EEG.T7"), data.get("EEG.Pz"), data.get("EEG.T8"), data.get("EEG.AF4")]
    else:
        return None



# Load model and scaler once
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

# Use columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    eeg_af3 = st.number_input("EEG.AF3", value=0.0, format="%.5f", step=0.001)
    eeg_pz = st.number_input("EEG.Pz", value=0.0, format="%.5f", step=0.001)

with col2:
    eeg_t7 = st.number_input("EEG.T7", value=0.0, format="%.5f", step=0.001)
    eeg_t8 = st.number_input("EEG.T8", value=0.0, format="%.5f", step=0.001)

with col3:
    eeg_af4 = st.number_input("EEG.AF4", value=0.0, format="%.5f", step=0.001)
    st.write("") 
    
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



if st.button("🔍 Predict"):
    features = np.array([[eeg_af3, eeg_t7, eeg_pz, eeg_t8, eeg_af4]])
    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = max(proba)

    label = "🤥 Lie" if prediction == 1 else "✅ Truth"

    # Show results with colors
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
