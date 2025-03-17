import streamlit as st
import pandas as pd
from agent import fetch_latest_data
from llm import query_chatbot

st.set_page_config(page_title="Smart Irrigation AI Dashboard", layout="wide")

page = st.sidebar.selectbox("Select Page", ["Dashboard", "Chatbot"])

if page == "Dashboard":
    st.title("📊 Monitoring Dashboard")
    latest_sensor_data = fetch_latest_data(SENSOR_TOPIC)
    if latest_sensor_data:
        df = pd.DataFrame([latest_sensor_data])
        st.dataframe(df)

elif page == "Chatbot":
    st.title("💬 AI Chatbot")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        response = query_chatbot(user_input)
        st.write("🤖 AI Response:", response)
