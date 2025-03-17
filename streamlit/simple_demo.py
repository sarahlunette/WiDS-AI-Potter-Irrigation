import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

# === Configuration ===
VECTOR_DB_PATH = "./vectorstore/index"
PDF_FOLDER_PATH = "../data/llm/documents"
MODEL_PATH = "./model_/gemma/"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# === Load Model ===
@st.cache_resource()
def load_model():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return tokenizer, model

# === Load PDFs into FAISS Vector Store ===
@st.cache_resource()
def load_pdfs_to_vectorstore(pdf_folder, vector_db_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(vector_db_path)
    return vector_db

# Initialize model and vector store
tokenizer, model = load_model()
vector_db = load_pdfs_to_vectorstore(PDF_FOLDER_PATH, VECTOR_DB_PATH)
memory = ConversationBufferMemory()

hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# === Streamlit Chatbot ===
st.title("💬 PDF RAG Chatbot")
user_input = st.text_input("Your question:")
if st.button("Ask"):
    response = qa_chain.invoke({"question": user_input, "chat_history": memory.load_memory_variables({}).get("history", [])})
    st.write("🤖 AI Response:", response)