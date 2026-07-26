from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage , HumanMessage , BaseMessage
from typing import TypedDict , Annotated 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

load_dotenv()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

primary = ChatGroq(model="openai/gpt-oss-120b")
fallback1 = ChatGroq(model="llama-3.3-70b-versatile")
fallback2 = ChatGroq(model="qwen/qwen3-32b")

model = primary.with_fallbacks([fallback1,fallback2])

def chat_node(state:ChatState):
    
    messages = state['messages']
    res = model.invoke(messages)

    return {'messages' : [res]}

connection = sqlite3.connect(database = 'venomind.db' , check_same_thread=False)

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")

connection.commit()

checkpointer = SqliteSaver(conn = connection)

builder = StateGraph(ChatState)

builder.add_node('chat_node' , chat_node)

builder.add_edge(START , 'chat_node')
builder.add_edge('chat_node' , END)

chatbot = builder.compile(checkpointer = checkpointer)

def save_conversation(thread_id, title="New Conversation"):
    cursor.execute("""
        INSERT OR IGNORE INTO conversations(thread_id, title)
        VALUES(?, ?)
    """, (str(thread_id), title))
    connection.commit()


def update_title(thread_id, title):
    cursor.execute("""
        UPDATE conversations
        SET title=?
        WHERE thread_id=?
    """, (title, str(thread_id)))
    connection.commit()


def retrive_all_threads():
    cursor.execute("""
        SELECT thread_id
        FROM conversations
        ORDER BY rowid DESC
    """)
    return [row[0] for row in cursor.fetchall()]


def retrive_all_titles():
    cursor.execute("""
        SELECT thread_id,title FROM conversations
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}