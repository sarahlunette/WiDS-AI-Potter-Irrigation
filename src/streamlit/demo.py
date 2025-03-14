import streamlit as st
import pandas as pd
import plotly.express as px
from kafka import KafkaConsumer, KafkaProducer
import json
from src.data.weather.weather import get_weather
from src.inference import run_algorithm  # Import the algorithm

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
SENSOR_TOPIC = "sensor_data"
FORECAST_TOPIC = "forecast_data"
COMMAND_TOPIC = "valve_commands"

# Weather API Configuration
API_KEY = "6a210d4c91e9f455ade17f75196c1a17"  # Replace with your actual API key

# Streamlit App with Multi-Page Navigation
st.set_page_config(page_title="Smart Irrigation AI Dashboard", layout="wide")

def fetch_data(lat, lon, api_key, soil_moisture, irrigation_schedule, manual_notes):
    # Fetch weather data
    weather_data = get_weather(lat, lon, api_key)
    
    if weather_data:
        # Combine weather data with other inputs
        data = {
            "soil_moisture": soil_moisture,
            "irrigation_schedule": irrigation_schedule,
            "manual_notes": manual_notes,
            "current_temperature": weather_data['current']['temperature'],
            "current_conditions": weather_data['current']['conditions'],
            "next_day_temperature_min": weather_data['next_day']['temperature_min'],
            "next_day_temperature_max": weather_data['next_day']['temperature_max'],
            "next_day_conditions": weather_data['next_day']['conditions']
        }
        return data
    else:
        st.error("Failed to retrieve weather data.")
        return None

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Farmer's Input", "Monitoring Dashboard", "Weather", "Chatbot"])

if page == "Farmer's Input":
    st.title("🚜 Farmer's Input")
    st.write("Enter manual data to enhance AI predictions.")
    
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=50)
    irrigation_schedule = st.selectbox("Irrigation Schedule", ["Morning", "Afternoon", "Evening"])
    manual_notes = st.text_area("Additional Notes")
    
    lat = st.text_input("Latitude", "37.7749")  # Default to San Francisco latitude
    lon = st.text_input("Longitude", "-122.4194")  # Default to San Francisco longitude
    
    if st.button("Submit Data"):
        data = fetch_data(lat, lon, API_KEY, soil_moisture, irrigation_schedule, manual_notes)
        if data:
            result = run_algorithm(data)  # Call the algorithm with the combined data
            st.success("Data submitted successfully!")
            st.json(result)

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

elif page == "Weather":
    st.title("🌤️ Weather Information")
    
    # Input latitude and longitude
    lat = st.text_input("Latitude", "37.7749")  # Default to San Francisco latitude
    lon = st.text_input("Longitude", "-122.4194")  # Default to San Francisco longitude
    
    if st.button("Get Weather"):
        weather_data = get_weather(lat, lon, API_KEY)
        
        if weather_data:
            st.subheader("Current Weather")
            st.write(f"Temperature: {weather_data['current']['temperature']}°C")
            st.write(f"Conditions: {weather_data['current']['conditions']}")
            
            st.subheader("Next Day Forecast")
            st.write(f"Min Temperature: {weather_data['next_day']['temperature_min']}°C")
            st.write(f"Max Temperature: {weather_data['next_day']['temperature_max']}°C")
            st.write(f"Conditions: {weather_data['next_day']['conditions']}")
        else:
            st.error("Failed to retrieve weather data.")

elif page == "Chatbot":
    st.title("💬 AI Chatbot")
    st.write("Ask the AI assistant about irrigation recommendations!")
    
    user_input = st.text_input("Your question:")
    
    if st.button("Ask"):
        response = "🤖 AI Response: This is a sample response. Connect to AI model here."
        st.write(response)
