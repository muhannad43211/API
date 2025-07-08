# streamlit_app.py
import streamlit as st
import requests

# ✅ FastAPI deployed endpoint
FASTAPI_URL = "https://api-nx09.onrender.com/predict"

# 🏷️ App title
st.title("⚽ Player Value Prediction")

# 📥 Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, value=25)
appearance = st.number_input("Appearance", min_value=0, max_value=500, value=50)
goals = st.number_input("Goals", min_value=0, max_value=100, value=20)
minutes_played = st.number_input("Minutes Played", min_value=0, max_value=5000, value=1500)
price_category = st.selectbox("Price Category", options=["Premium", "Mid", "Budget"])

# 🧾 Input data formatted as JSON
input_data = {
    "age": age,
    "appearance": appearance,
    "goals": goals,
    "minutes_played": minutes_played,
    "price_category": price_category,
    "Highest_valuated_price_euro": 0.0  # Extra field to match model input schema
}

# ⏩ Button to send prediction request
if st.button("Predict Player Value"):
    try:
        response = requests.post(FASTAPI_URL, json=input_data)

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"💰 Predicted Value: {prediction['pred']} Euro")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚫 Connection Error: {e}")
