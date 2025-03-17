import os
import json
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
import threading
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# TODO: add historical irrigation data and decision and crop yields

# === Import Custom Functions ===
import sys
sys.path.append(os.path.abspath('..'))
from src.websearch.search_engine import get_web_results  # Web search API
from src.forecast.generate_forecast import generate_forecast  # GAN Forecasting
from src.llm.llm_rag_feed import get_weather  # Weather function

st.set_page_config(page_title="Smart Irrigation AI Dashboard", layout="wide")

# === Configuration ===
KAFKA_BROKER = "localhost:9092"
SENSOR_TOPIC = "sensor_data"
FORECAST_TOPIC = "forecast_data"
COMMAND_TOPIC = "valve_commands"
ENRICHED_DATA_TOPIC = "enriched_data_topic"
VECTOR_DB_PATH = "./vectorstore/index"  # Persistent FAISS storage
PDF_FOLDER_PATH = "../data/llm/documents"
MODEL_PATH = "./model_/gemma/"  # Persistent model storage

# Ensure directories exist
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# === Load Gemma Model (Persistent) ===
@st.cache_resource()
def load_gemma_model():
    model_id = "google/gemma-2b"
    
    if not os.path.exists(MODEL_PATH):
        print("🆕 Downloading and saving Gemma model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    else:
        print("🔄 Loading saved Gemma model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    return tokenizer, model

# === Load PDFs into FAISS Vector Store (Persistent) ===
@st.cache_resource()
def load_pdfs_to_vectorstore(pdf_folder, vector_db_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(os.path.join(vector_db_path, "index")):
        print("🔄 Loading existing FAISS vector store...")
        return FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("🆕 Creating new FAISS vector store...")
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(vector_db_path)  # Save FAISS to disk
    return vector_db

# Initialize model and vector store
tokenizer, model = load_gemma_model()
vector_db = load_pdfs_to_vectorstore(PDF_FOLDER_PATH, VECTOR_DB_PATH)
memory = ConversationBufferMemory()

# Create LLM pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# === Conversational AI Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# === Custom RAG Query Function (For Questions & Sensor Data) ===
def custom_rag_query(question, sensor_data=None):
    location = sensor_data.get("location", "Bordeaux") if sensor_data else "Bordeaux"
    weather_data = get_weather(location)
    forecast_data = generate_forecast(sensor_data) if sensor_data else "No sensor data provided."
    
    query_input = {
        "question": f"""
        📝 **User Question:** {question}
        📡 **Sensor Data:** {sensor_data if sensor_data else 'N/A'}
        🌦 **Weather:** {weather_data}
        🔮 **Forecast:** {forecast_data}
        """,
        "chat_history": memory.load_memory_variables({}).get("history", [])
    }
    
    response = qa_chain.run(query_input)
    return response

# === Sensor Data Processing Function (For Dashboard) ===
def process_sensor_data(sensor_data):
    location = sensor_data.get("location", "Bordeaux")
    weather_data = get_weather(location)
    forecast_data = generate_forecast(sensor_data)  # Assuming sensor_data is already enriched
    
    query_input = {
        "question": f"""
        📡 **Sensor Data:** {sensor_data}
        🌦 **Weather:** {weather_data}
        🔮 **Forecast:** {forecast_data}
        
        Based on the above data, **output an irrigation control value between 1 (dry) and 10 (wet)**.
        """,
        "chat_history": memory.load_memory_variables({}).get("history", [])
    }
    
    response = qa_chain.run(query_input)
    try:
        return float(response.strip())  # Convert AI output to numerical value
    except ValueError:
        print("⚠️ AI returned an invalid value, defaulting to 5")
        return 5  # Default neutral value

# === Automated AI Agent (Continuous Scale 1-10) ===
def automated_decision_making():
    consumer = KafkaConsumer(
        SENSOR_TOPIC, bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    for message in consumer:
        sensor_data = message.value
        control_value = process_sensor_data(sensor_data)  # AI-generated irrigation control level
        command = {"sector": sensor_data['sector'], "control_value": control_value}
        producer.send(COMMAND_TOPIC, command)
        print(f"✅ Sent control command: {command}")

# === Streamlit App ===
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Weather", "Chatbot"])

if page == "Chatbot":
    st.title("💬 AI Chatbot")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        response = custom_rag_query(user_input)
        st.write("🤖 AI Response:", response)
