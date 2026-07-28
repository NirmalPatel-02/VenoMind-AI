import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = os.getenv("DB_PATH", "venomind.db")

ACTIVE_SET = 1

if ACTIVE_SET == 1:
    primary_model = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )
    title_primary = primary_model
    title_fallback = llm

elif ACTIVE_SET == 2:
    primary_model = ChatOpenAI(
        model="openrouter/free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )
    llm = ChatOpenAI(
        model="openrouter/free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )
    title_primary = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )
    title_fallback = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

elif ACTIVE_SET == 3:
    primary_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.7,
        max_retries=2
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_retries=2
    )
    title_primary = primary_model
    title_fallback = llm

elif ACTIVE_SET == 4:
    primary_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.7,
        max_retries=2
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_retries=2
    )
    title_primary = llm
    title_fallback = llm

elif ACTIVE_SET == 5:
    primary_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    title_primary = llm
    title_fallback = llm