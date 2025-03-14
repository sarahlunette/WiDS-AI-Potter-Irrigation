import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# === 1️⃣ Load Gemma Model ===
@st.cache_resource()
def load_gemma_model():
    model_id = "google/gemma-2b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model

# === 2️⃣ Load PDFs and Store in FAISS ===
@st.cache_resource()
def load_pdfs_to_vectorstore(pdf_folder):
    documents = []
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Convert to embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(texts, embeddings)
    
    return vector_db

# === 3️⃣ Initialize Model and Vector Store ===
tokenizer, model = load_gemma_model()

# Create a pipeline for text generation
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=512
)

# Wrap in LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=hf_pipeline)

vector_db = load_pdfs_to_vectorstore("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/data/llm/documents/")
memory = ConversationBufferMemory()

# === 4️⃣ Create the Conversational Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# === 5️⃣ Chatbot Function ===
chat_history = []  # Store conversation history

def chat_with_gemma(question):
    global chat_history

    # Invoke the QA chain with question & history
    response = qa_chain.invoke({"question": question, "chat_history": chat_history})

    # Update history
    chat_history.append((question, response["answer"]))

    return response["answer"]

# === 6️⃣ Streamlit Chat Interface ===
st.title("🤖 Gemma Chatbot with Document Retrieval")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if user_input:
    response = chat_with_gemma(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Gemma", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    st.text(f"{sender}: {message}")

