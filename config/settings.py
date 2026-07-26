import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.getenv("DB_PATH", "venomind.db")

primary_model = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.7,
    max_retries=2
)

fallback_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
llm = primary_model.with_fallbacks([fallback_model])