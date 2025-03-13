import streamlit as st
import pandas as pd
import plotly.express as px
from kafka import KafkaConsumer, KafkaProducer
import json

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
SENSOR_TOPIC = "sensor_data"
FORECAST_TOPIC = "forecast_data"
COMMAND_TOPIC = "valve_commands"

# Streamlit App with Multi-Page Navigation
st.set_page_config(page_title="Smart Irrigation AI Dashboard", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Farmer's Input", "Monitoring Dashboard", "Chatbot"])

if page == "Farmer's Input":
    st.title("🚜 Farmer's Input")
    st.write("Enter manual data to enhance AI predictions.")
    
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=50)
    irrigation_schedule = st.selectbox("Irrigation Schedule", ["Morning", "Afternoon", "Evening"])
    manual_notes = st.text_area("Additional Notes")
    
    if st.button("Submit Data"):
        st.success("Data submitted successfully!")

elif page == "Monitoring Dashboard":
    st.title("📊 Monitoring Dashboard")
    
    # Kafka Consumer for Sensor Data
    consumer = KafkaConsumer(
        SENSOR_TOPIC, bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
    
    data = []
    for message in consumer:
        data.append(message.value)
        if len(data) > 50:  # Limit the displayed data
            break
    
    df = pd.DataFrame(data)
    
    sensor_list = df["sector"].unique() if not df.empty else []
    selected_sensor = st.selectbox("Select Sensor", sensor_list)
    
    variable = st.selectbox("Select Variable", ["soil_moisture", "temperature", "humidity"])
    
    if not df.empty:
        fig = px.line(df[df["sector"] == selected_sensor], x=df.index, y=variable, title=f"{variable} Over Time")
        st.plotly_chart(fig)
    
elif page == "Chatbot":
    st.title("💬 AI Chatbot")
    st.write("Ask the AI assistant about irrigation recommendations!")
    
    user_input = st.text_input("Your question:")
    
    if st.button("Ask"):
        response = "🤖 AI Response: This is a sample response. Connect to AI model here."
        st.write(response)
