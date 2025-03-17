import os
import sys
sys.path.append(os.path.abspath('..'))
from websearch.search_engine import get_web_results
from forecast.generate_forecast import generate_forecast
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTOR_DB_PATH = "vectorstore/index"
PDF_FOLDER_PATH = "../data/llm/documents"
QA_FILE_PATH = "../data/llm/QR/Q&A.xlsx"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def load_gemma_model():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/model/gemma/")
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/model/gemma/", torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model


def load_vectorstore(pdf_folder, qa_file, vector_db_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(vector_db_path):
        return FAISS.load_local(vector_db_path, embeddings)
    
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            documents.extend(PyPDFLoader(os.path.join(pdf_folder, pdf_file)).load())
    
    if os.path.exists(qa_file):
        df = pd.read_excel(qa_file)
        documents.extend([f"{q} {a}" for q, a in zip(df["Question"].astype(str), df["Answer"].astype(str))])
    
    texts = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100).split_documents(documents)
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(vector_db_path)
    return vector_db

# Load models and vector store
tokenizer, model = load_gemma_model()
vector_db = load_vectorstore(PDF_FOLDER_PATH, QA_FILE_PATH, VECTOR_DB_PATH)
memory = ConversationBufferMemory()

hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

def query_chatbot(user_input):
    latest_sensor_data = fetch_latest_data(SENSOR_TOPIC)
    forecast_data = generate_forecast(latest_sensor_data)
    websearch_data = get_web_results(user_input)
    
    enriched_prompt = f"Latest sensor data: {latest_sensor_data}. Forecast: {forecast_data}. Web search results: {websearch_data}.\n{user_input}"
    return qa_chain.run({"question": enriched_prompt, "chat_history": memory.load_memory_variables({}).get("history", [])})

