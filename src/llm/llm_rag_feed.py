import os
import requests
import json
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

IOT_API_URL = "http://localhost:5000/sensor_data"  # IoT API (replace with real)
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
PAST_WEATHER_API = "http://localhost:5001/past_weather"  # GAN model past data
FORECAST_API = "http://localhost:5002/weather_forecast"  # GAN forecast system

# ✅ Step 1: Load PDFs into FAISS VectorDB
def load_pdf_to_vectorstore(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    vector_db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    return vector_db

vector_db = load_pdf_to_vectorstore("water_policies.pdf")

# ✅ Step 2: Load AI Model & Memory
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory()
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)

# ✅ Step 3: Fetch IoT Sensor Data # TODO: fetch with kafka
def get_iot_data():
    try:
        response = requests.get(IOT_API_URL)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"IoT API Error: {e}")
    return {}

# ✅ Step 4: Fetch Current Weather Data
def get_weather(city="Bordeaux"):
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        if response.status_code == 200:
            weather_data = response.json()
            return {
                "temperature": weather_data["main"]["temp"],
                "humidity": weather_data["main"]["humidity"],
                "conditions": weather_data["weather"][0]["description"]
            }
    except Exception as e:
        print(f"Weather API Error: {e}")
    return {}

# ✅ Step 5: Fetch Past Weather Data (from GAN System)
def get_past_weather():
    try:
        response = requests.get(PAST_WEATHER_API)
        if response.status_code == 200:
            return response.json()  # Expected: {"temp_avg": 25.1, "rain_avg": 3.2, "humidity_avg": 65}
    except Exception as e:
        print(f"Past Weather API Error: {e}")
    return {}

# ✅ Step 6: Fetch Weather Forecast Data (from GAN System)
def get_forecast():
    try:
        response = requests.get(FORECAST_API)
        if response.status_code == 200:
            return response.json()  # Expected: {"temp_pred": 28.4, "rain_pred": 5.0, "humidity_pred": 60}
    except Exception as e:
        print(f"Forecast API Error: {e}")
    return {}

# ✅ Step 7: Process Query with RAG + External Data
def custom_rag_query(user_query):
    # Get real-time data
    iot_data = get_iot_data()
    weather_data = get_weather()
    past_weather = get_past_weather()
    forecast_data = get_forecast()

    # Construct dynamic context
    external_context = f"""
    📡 **Current Sensor Data:**
    - Sector: {iot_data.get("sector", "Unknown")}
    - Soil Moisture: {iot_data.get("moisture", "N/A")}%
    - Temperature: {iot_data.get("temperature", "N/A")}°C

    🌦 **Current Weather:**
    - Temperature: {weather_data.get("temperature", "N/A")}°C
    - Humidity: {weather_data.get("humidity", "N/A")}%
    - Conditions: {weather_data.get("conditions", "N/A")}

    🔙 **Past Weather Data (GAN Model):**
    - Avg Temp: {past_weather.get("temp_avg", "N/A")}°C
    - Avg Rain: {past_weather.get("rain_avg", "N/A")} mm
    - Avg Humidity: {past_weather.get("humidity_avg", "N/A")}%

    🔮 **Forecast Data (GAN Prediction):**
    - Predicted Temp: {forecast_data.get("temp_pred", "N/A")}°C
    - Predicted Rain: {forecast_data.get("rain_pred", "N/A")} mm
    - Predicted Humidity: {forecast_data.get("humidity_pred", "N/A")}%

    🚜 **Farmer's Question:** {user_query}

    Provide an answer considering water policies, IoT data, weather trends, past records, and forecast predictions.
    """

    # Run query through RAG pipeline
    response = qa_chain.run(external_context)
    return response

# ✅ Step 8: Run Chatbot
if __name__ == "__main__":
    user_query = "Should I water my vineyard today?"
    response = custom_rag_query(user_query)

    print(f"👨‍🌾 Farmer: {user_query}")
    print(f"🤖 AI: {response}")
