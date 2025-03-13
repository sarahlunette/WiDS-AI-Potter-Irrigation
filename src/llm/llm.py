from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load LLM
llm = ChatOpenAI(model_name="gpt-4", openai_api_key="YOUR_OPENAI_API_KEY")

# Knowledge Base (Water Policies, Past Data)
data = [
    "Watering is restricted from 10 AM to 6 PM in summer.",
    "Ideal soil moisture for grapevines is 25-35%.",
    "Water prices are expected to increase by 15% next month."
]
vector_db = FAISS.from_texts(data, OpenAIEmbeddings())

# Chatbot with Memory
memory = ConversationBufferMemory()
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# Farmer Interaction
query = "What are the watering restrictions in summer?"
response = qa_chain.run(query)
print(f"👨‍🌾 Farmer: {query}")
print(f"🤖 AI: {response}")
