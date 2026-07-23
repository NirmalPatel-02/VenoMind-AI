from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage , HumanMessage , BaseMessage
from typing import TypedDict , Annotated 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

primary = ChatGroq(
    model="openai/gpt-oss-120b"
)

fallback1 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

fallback2 = ChatGroq(
    model="qwen/qwen3-32b"
)

model = primary.with_fallbacks([fallback1,fallback2])

def chat_node(state:ChatState):
    
    messages = state['messages']
    res = model.invoke(messages)

    return {'messages' : [res]}

checkpointer = InMemorySaver()

builder = StateGraph(ChatState)

builder.add_node('chat_node' , chat_node)

builder.add_edge(START , 'chat_node')
builder.add_edge('chat_node' , END)

chatbot = builder.compile(checkpointer = checkpointer)