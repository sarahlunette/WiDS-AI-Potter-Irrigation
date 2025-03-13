import streamlit as st

st.title("Smart Irrigation AI Dashboard")

soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 50)
weather_forecast = "Rain in 2 days"

st.write(f"📢 Weather Forecast: {weather_forecast}")
st.write(f"💧 Recommended Irrigation: {'Water' if soil_moisture < 30 else 'Hold'}")
