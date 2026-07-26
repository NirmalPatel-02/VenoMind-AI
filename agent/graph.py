import sqlite3
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import llm, DB_PATH
from agent.state import ChatState

def chat_node(state: ChatState):
    """Main LLM response node."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=db_conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

chatbot = builder.compile(checkpointer=checkpointer)

def generate_title(first_message: str) -> str:
    """Utility function to auto-generate thread titles based on first message."""
    prompt = [
        SystemMessage(content="""
            Generate a concise, high-level summary title for this conversation.
            Rules:
            - Maximum 5 words.
            - Do not use quotes or special characters.
            - Do not add punctuation.
            - Output ONLY the title text.
        """),
        HumanMessage(content=first_message)
    ]
    response = llm.invoke(prompt)
    return response.content.strip()