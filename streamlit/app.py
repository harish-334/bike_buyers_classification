import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import base64

# -----------------------------
# Page config (MUST be first)
# -----------------------------
st.set_page_config(
    page_title="Bike Buyers Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "global_best_model.pkl"
UI_CONFIG_PATH = BASE_DIR / "streamlit" / "ui_config.json"
BG_IMAGE_PATH = Path(__file__).parent / "assets" / "bike_bg.jpg"

# -----------------------------
# Background + Global CSS
# -----------------------------
def add_bg_from_local(image_path: Path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Remove Streamlit default UI */
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stApp > header {{display: none;}}

        /* Background */
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main container spacing */
        .block-container {{
            padding-top: 2rem;
        }}

        /* Glass card */
        .glass {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 32px;
            border: 1px solid rgba(255,255,255,0.25);
            margin-top: 24px;
        }}

        /* Text color */
        h1, h2, h3, label, p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local(BG_IMAGE_PATH)

# -----------------------------
# Load model & UI config
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_ui_config():
    with open(UI_CONFIG_PATH) as f:
        return json.load(f)

model = load_model()
ui_config = load_ui_config()

# -----------------------------
# Header / Hero section
# -----------------------------
st.title("🚲 Bike Purchase Intelligence")
st.markdown(
    "AI-powered prediction system to estimate **bike buying likelihood** "
    "based on customer lifestyle and demographics."
)

# -----------------------------
# Glass Card (ONLY ONE)
# -----------------------------

with st.form("prediction_form"):
    st.subheader("Customer Details")

    input_data = {}

    # Categorical inputs
    for col, options in ui_config["categorical_features"].items():
        input_data[col] = st.selectbox(
            col.replace("_", " ").title(),
            options
        )

    # Numeric inputs
    for col, bounds in ui_config["numeric_features"].items():
        input_data[col] = st.slider(
            col.replace("_", " ").title(),
            int(bounds["min"]),
            int(bounds["max"])
        )

    submitted = st.form_submit_button("🔍 Predict")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction Output
# -----------------------------
import requests

API_URL = "http://api:8000/predict"

if submitted:
    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probability = result["probability"]

        st.subheader("Prediction Result")

        # Confidence bar
        st.progress(int(probability * 100))

        if prediction == 1:
            st.success(
                f"✅ Likely to buy a bike\n\n"
                f"**Confidence:** {probability*100:.1f}%"
            )
        else:
            st.warning(
                f"❌ Unlikely to buy a bike\n\n"
                f"**Confidence:** {(1-probability)*100:.1f}%"
            )

        # Explanation text
        if probability > 0.75:
            st.info("🔍 Strong confidence prediction")
        elif probability > 0.55:
            st.info("🔍 Moderate confidence prediction")
        else:
            st.info("🔍 Low confidence - borderline case")

    else:
        st.error("API error. Please try again.")
