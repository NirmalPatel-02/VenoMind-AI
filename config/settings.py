import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = os.getenv("DB_PATH", "venomind.db")

gemini_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.7
)

groq_agent_70b = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_retries=2
)

groq_agent_8b = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_retries=2
)

primary_model = gemini_agent.with_fallbacks([groq_agent_70b, groq_agent_8b])

groq_title_primary = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

groq_title_qwen = ChatGroq(
    model="qwen-2.5-32b",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

gemini_title_fallback = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.5
)

llm = groq_title_primary.with_fallbacks([groq_title_qwen, gemini_title_fallback])