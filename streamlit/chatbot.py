import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import sys
sys.path.append(os.path.abspath('..'))
from src.websearch.search_engine import get_web_results  # Web search API
from src.forecast.generate_forecast import generate_forecast  # GAN Forecasting
from src.llm.llm_rag_feed import load_pdfs_to_vectorstore, get_weather  # Import relevant functions

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load Vector DB
vector_db = load_pdfs_to_vectorstore("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents")

# Load AI Model & Memory
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory()
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

def custom_rag_query(sensor_data):
    weather_data = get_weather(sensor_data.get("location", "Bordeaux"))
    forecast_data = generate_forecast(sensor_data)
    web_results = get_web_results("irrigation best practices")
    
    external_context = f"""
    📡 **Current Sensor Data:**
    - Sector: {sensor_data.get("sector", "Unknown")}
    - Soil Moisture: {sensor_data.get("soil_moisture", "N/A")}%
    - Temperature: {sensor_data.get("temperature", "N/A")}°C
    - Humidity: {sensor_data.get("humidity", "N/A")}%

    🌦 **Current Weather:**
    - Temperature: {weather_data.get("temperature", "N/A")}°C
    - Humidity: {weather_data.get("humidity", "N/A")}%
    - Conditions: {weather_data.get("conditions", "N/A")}

    🔮 **Forecast Data (GAN Prediction):**
    - Predicted Irrigation: {forecast_data.get("predicted_irrigation", "N/A")}
    """
    
    response = qa_chain.run(external_context)
    return response

# Streamlit UI
st.title("Smart Irrigation Chatbot")
st.write("Ask about irrigation, weather, and farming recommendations!")

user_input = st.text_input("Enter your query:")

if user_input:
    sensor_data = {"location": "Bordeaux", "sector": "Vineyard", "soil_moisture": 30, "temperature": 22, "humidity": 60}
    response = custom_rag_query(sensor_data)
    st.write("🤖 AI Response:", response)
