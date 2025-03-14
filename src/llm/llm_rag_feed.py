import os
import sys
import requests
import json
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from kafka import KafkaConsumer
from src.websearch.search_engine import get_web_results  # Web search API
from src.forecast.generate_forecast import generate_forecast  # GAN Forecasting

# Add parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API_KEY = os.getenv("OWM_API_KEY")

ENRICHED_DATA_TOPIC = "enriched_data_topic"
KAFKA_BROKER = "localhost:9092"

# ✅ Step 1: Load PDFs into FAISS VectorDB
def load_pdfs_to_vectorstore(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    vector_db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    return vector_db

vector_db = load_pdfs_to_vectorstore("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents") # TODO: Change path

# ✅ Step 2: Load AI Model & Memory
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory()
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# ✅ Step 3: Kafka Consumer Setup
def consume_kafka_messages():
    consumer = KafkaConsumer(
        ENRICHED_DATA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    for message in consumer:
        sensor_data = message.value
        print(f"Received Sensor Data: {sensor_data}")
        response = custom_rag_query(sensor_data)
        print(f"🤖 AI Response: {response}")

# ✅ Step 4: Fetch Current Weather Data
def get_weather(city="Bordeaux"):
    params = {"q": city, "appid": OWM_API_KEY, "units": "metric"}
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params)
        if response.status_code == 200:
            weather_data = response.json()
            return {
                "temperature": weather_data["main"]["temp"],
                "humidity": weather_data["main"]["humidity"],
                "conditions": weather_data["weather"][0]["description"]
            }
    except Exception as e:
        print(f"Weather API Error: {e}")
    return {}

# ✅ Step 5: Fetch Irrigation Forecast Data (from GAN System)
def get_forecast(sensor_data):
    return generate_forecast(sensor_data)

# ✅ Step 6: Process Query with RAG + External Data
def custom_rag_query(sensor_data):
    weather_data = get_weather(sensor_data.get("location", "Bordeaux"))
    forecast_data = get_forecast(sensor_data)
    web_results = get_web_results("irrigation best practices")
    
    external_context = f"""
    📡 **Current Sensor Data:**
    - Sector: {sensor_data.get("sector", "Unknown")}
    - Soil Moisture: {sensor_data.get("soil_moisture", "N/A")}%
    - Temperature: {sensor_data.get("temperature", "N/A")}°C
    - Humidity: {sensor_data.get("humidity", "N/A")}%
    - Evapotranspiration: {sensor_data.get("evapotranspiration", "N/A")}

    🌦 **Current Weather:**
    - Temperature: {weather_data.get("temperature", "N/A")}°C
    - Humidity: {weather_data.get("humidity", "N/A")}%
    - Conditions: {weather_data.get("conditions", "N/A")}

    🔮 **Forecast Data (GAN Prediction):**
    - Predicted Irrigation: {forecast_data.get("predicted_irrigation", "N/A")}
    - Sensor Data Used:
      - Temperature: {forecast_data["sensor_data_used"].get("temperature", "N/A")}°C
      - Humidity: {forecast_data["sensor_data_used"].get("humidity", "N/A")}%
      - Soil Moisture: {forecast_data["sensor_data_used"].get("soil_moisture", "N/A")}%
      - Solar Radiation: {forecast_data["sensor_data_used"].get("solar_radiation", "N/A")} W/m²

    🚜 **Farmer's Query:** Should I water my vineyard today?

    Provide an answer considering water policies, IoT data, weather trends, and forecast predictions.
    """
    
    response = qa_chain.run(external_context)
    return response

# ✅ Step 7: Start Kafka Consumer
if __name__ == "__main__":
    consume_kafka_messages()
