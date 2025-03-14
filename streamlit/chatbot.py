import os
import json
import torch
import streamlit as st
from kafka import KafkaConsumer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.websearch.search_engine import get_web_results  # Web search API
from src.forecast.generate_forecast import generate_forecast  # GAN Forecasting
from src.llm.llm_rag_feed import get_weather  # Import weather function

# === 1️⃣ Load Environment Variables ===
KAFKA_BROKER = "localhost:9092"
ENRICHED_DATA_TOPIC = "enriched_data_topic"
VECTOR_DB_PATH = "vectorstore/index"
PDF_FOLDER_PATH = "/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents"

# === 2️⃣ Load Gemma Model ===
@st.cache_resource()
def load_gemma_model():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

# === 3️⃣ Load PDFs and Store in FAISS ===
@st.cache_resource()
def load_pdfs_to_vectorstore(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db

# === 4️⃣ Initialize Model and Vector Store ===
tokenizer, model = load_gemma_model()
vector_db = load_pdfs_to_vectorstore(PDF_FOLDER_PATH)
memory = ConversationBufferMemory()

# Create a pipeline for text generation
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# === 5️⃣ Create the Conversational Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# === 6️⃣ Custom RAG Query Function ===
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

# === 7️⃣ Kafka Consumer for Real-Time Data ===
def consume_kafka_messages():
    consumer = KafkaConsumer(
        ENRICHED_DATA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        return message.value  # Get latest sensor data

# === 8️⃣ Streamlit Chat Interface ===
st.title("🤖 Smart Irrigation Chatbot with Gemma & FAISS")
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
