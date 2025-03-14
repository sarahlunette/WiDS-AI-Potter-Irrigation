import os
import streamlit as st
import json
from kafka import KafkaConsumer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.websearch.search_engine import get_web_results  # Web search API
from src.forecast.generate_forecast import generate_forecast  # GAN Forecasting
from src.llm.llm_rag_feed import load_pdfs_to_vectorstore, get_weather  # Import relevant functions
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
KAFKA_BROKER = "localhost:9092"
ENRICHED_DATA_TOPIC = "enriched_data_topic"

# Load Vector DB
#vector_db = load_pdfs_to_vectorstore("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents")

# Load Open-Source AI Model & Memory
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
memory = ConversationBufferMemory()

VECTOR_DB_PATH = "vectorstore/index"

def load_pdfs_to_vectorstore(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # Generate embeddings using a lightweight open-source model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if the FAISS index already exists
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing FAISS vector store...")
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    else:
        print("Creating a new FAISS vector store...")
        vector_db = FAISS.from_documents(split_docs, embeddings)
        vector_db.save_local(VECTOR_DB_PATH)  # Save it permanently

    return vector_db

# Call function to load PDFs into FAISS
vector_db = load_pdfs_to_vectorstore("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents")

qa_chain = ConversationalRetrievalChain.from_llm(model, vector_db.as_retriever(), memory=memory)

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

def consume_kafka_messages():
    consumer = KafkaConsumer(
        ENRICHED_DATA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        return message.value  # Get latest sensor data
    
# Streamlit UI
st.title("Smart Irrigation Chatbot")
st.write("Ask about irrigation, weather, and farming recommendations!")

# Display real-time sensor data from Kafka
sensor_data = consume_kafka_messages()
if sensor_data:
    st.write("📡 Latest Sensor Data:", sensor_data)
    response = custom_rag_query(sensor_data)
    st.write("🤖 AI Response:", response)

# User manual query input
user_input = st.text_input("Enter your query:")
if user_input:
    manual_sensor_data = {"location": "Bordeaux", "sector": "Vineyard", "soil_moisture": 30, "temperature": 22, "humidity": 60}
    response = custom_rag_query(manual_sensor_data)
    st.write("🤖 AI Response:", response)
