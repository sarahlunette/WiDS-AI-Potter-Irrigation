import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Simple Chatbot", layout="wide")

# Load the model
@st.cache_resource()
def load_model():
    return pipeline("text-generation", model="google/gemma-2b")

model = load_model()

# Streamlit Chatbot
st.title("💬 Simple Chatbot")
user_input = st.text_input("Your question:")
if st.button("Ask"):
    response = model(user_input, max_new_tokens=100)[0]['generated_text']
    st.write("🤖 AI Response:", response)