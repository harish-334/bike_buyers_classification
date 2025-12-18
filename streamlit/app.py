import streamlit as st
import json
import requests
import base64
from pathlib import Path

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
BASE_DIR = Path(__file__).resolve().parent
UI_CONFIG_PATH = BASE_DIR / "ui_config.json"
BG_IMAGE_PATH = BASE_DIR / "assets" / "bike_bg.jpg"

# -----------------------------
# Background + CSS
# -----------------------------
def add_bg_from_local(image_path: Path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            .stApp > header {{display: none;}}

            .stApp {{
                background:
                    linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                    url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            .block-container {{
                padding-top: 2rem;
            }}

            h1, h2, h3, label, p {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # safe for cloud

add_bg_from_local(BG_IMAGE_PATH)

# -----------------------------
# Load UI config
# -----------------------------
@st.cache_data
def load_ui_config():
    with open(UI_CONFIG_PATH) as f:
        return json.load(f)

ui_config = load_ui_config()

# -----------------------------
# Header
# -----------------------------
st.title("🚲 Bike Purchase Intelligence")
st.markdown(
    "AI-powered prediction system to estimate **bike buying likelihood** "
    "based on customer lifestyle and demographics."
)

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):
    st.subheader("Customer Details")
    input_data = {}

    for col, options in ui_config["categorical_features"].items():
        input_data[col] = st.selectbox(col.replace("_", " ").title(), options)

    for col, bounds in ui_config["numeric_features"].items():
        input_data[col] = st.slider(
            col.replace("_", " ").title(),
            int(bounds["min"]),
            int(bounds["max"])
        )

    submitted = st.form_submit_button("🔍 Predict")

# -----------------------------
# API Call
# -----------------------------
API_URL = "https://bike-buyers-api.onrender.com/predict"

if submitted:
    with st.spinner("🔄 Predicting..."):
        try:
            response = requests.post(API_URL, json=input_data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]

                st.subheader("Prediction Result")
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
            else:
                st.error(f"❌ API error ({response.status_code})")

        except requests.exceptions.Timeout:
            st.error("⏳ API waking up. Try again in 10 seconds.")

        except requests.exceptions.RequestException:
            st.error("❌ Cannot reach prediction server.")
