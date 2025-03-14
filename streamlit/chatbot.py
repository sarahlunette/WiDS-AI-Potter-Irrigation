import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate import dispatch_model

# === 1️⃣ Load Gemma Model ===
def load_gemma_model():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with disk offloading
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    return tokenizer, model

# === 2️⃣ Load PDFs and Store in FAISS ===
def load_pdfs_to_vectorstore(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents.extend(pdf_loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Convert to embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db

# === 3️⃣ Initialize Model and Vector Store ===
tokenizer, model = load_gemma_model()

# Create a text-generation pipeline
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

# === 5️⃣ Run the Chatbot ===
def chat_with_gemma(question):
    response = qa_chain.invoke({"question": question})
    return response["answer"]

# Example
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chat_with_gemma(user_input)
        print("Gemma:", response)
