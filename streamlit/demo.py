import os
import json
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
VECTOR_DB_PATH = "vectorstore/index"
PDF_FOLDER_PATH = "./data/llm/documents"

# === Load Gemma Model ===
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

# === Load PDFs into FAISS Vector Store ===
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

# Initialize model and vector store
tokenizer, model = load_gemma_model()
vector_db = load_pdfs_to_vectorstore(PDF_FOLDER_PATH)
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

# === Automated AI Agent ===
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
        response = custom_rag_query(sensor_data)
        
        if sensor_data['soil_moisture'] < 30:  # Example rule
            command = {"sector": sensor_data['sector'], "action": "open_valve"}
            producer.send(COMMAND_TOPIC, command)

# === Custom RAG Query Function ===
def custom_rag_query(sensor_data):
    weather_data = get_weather(sensor_data.get("location", "Bordeaux"))
    forecast_data = generate_forecast(sensor_data)
    web_results = get_web_results("irrigation best practices")
    
    # Formulate the query input
    query_input = {
        "question": f"""
        📡 **Sensor Data:** {sensor_data}
        🌦 **Weather:** {weather_data}
        🔮 **Forecast:** {forecast_data}
        🌍 **Web Insights:** {web_results[:3]}
        """,
        "chat_history": memory.load_memory_variables({}).get("history", [])  # Ensure chat history is included
    }
    
    response = qa_chain.run(query_input)
    return response


# === Streamlit App ===
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Weather", "Chatbot"])

if page == "Dashboard":
    st.title("📊 Monitoring Dashboard")
    consumer = KafkaConsumer(SENSOR_TOPIC, bootstrap_servers=KAFKA_BROKER, value_deserializer=lambda m: json.loads(m.decode("utf-8")))
    data = [message.value for message in consumer]
    df = pd.DataFrame(data)
    st.dataframe(df)

elif page == "Weather":
    st.title("🌤️ Weather Information")
    lat, lon = st.text_input("Latitude", "37.7749"), st.text_input("Longitude", "-122.4194")
    if st.button("Get Weather"):
        weather_data = get_weather(lat, lon)
        st.write(weather_data)

elif page == "Chatbot":
    st.title("💬 AI Chatbot")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        response = custom_rag_query({"location": "Bordeaux", "sector": "Vineyard"})
        st.write("🤖 AI Response:", response)

# Start automated agent
automated_decision_making()
