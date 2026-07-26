import sqlite3
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import llm, DB_PATH, primary_model
from agent.state import ChatState , SearchInput

SYSTEM_PROMPT = SystemMessage(content="""You are VenoMind AI, an expert AI assistant.
Always format your responses using rich Markdown:
- Use code blocks with language identifiers for code.
- Use Markdown tables (| Header | Header |) for structured data or metrics.
- Use bold text and bullet points for readability.

STRICT REAL-TIME DATA RULE:
1. Whenever the user asks for stock prices, financial metrics, news, sports scores, or current event forecasts, you MUST invoke the `web_search` tool.
2. DO NOT guess, fabricate, or predict future stock prices (e.g. 2025-2050 price targets) out of memory. Base all numbers ONLY on the provided search results.
3. If User query demand accurate data then only give him accurate data dont predict or make assumption only give data if u are accurate about that
""")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Searches the web using DuckDuckGo to obtain up-to-date real-time news and facts."""
    try:
        search = DuckDuckGoSearchRun()
        optimized_query = rewrite_query(query)
        return search.run(optimized_query)
    except Exception as e:
        return f"Error executing search: {str(e)}"

tools = [web_search]

llm_with_tool = primary_model.bind_tools(tools, tool_choice="auto")

def rewrite_query(original_query: str) -> str:
    """Transforms conversational/bloated queries into concise search engine terms."""
    prompt = [
        SystemMessage(content="""You are a search query optimizer. 
Convert the user's input into 2-4 clean, highly targeted search keywords for a search engine.
Do not include fluff like 'future prospects', 'all news', or 'plz'.
Example Input: 'what is jio finance share price and also all news related to it and future about shre price plz'
Example Output: 'JIOFIN stock price live NSE BSE news'
Output ONLY the optimized query text."""),
        HumanMessage(content=original_query)
    ]
    response = llm.invoke(prompt)
    return response.content.strip()

def get_safe_history(messages: list[BaseMessage], max_messages=10) -> list[BaseMessage]:
    """Truncates message history safely without breaking tool call pairs or system prompt."""
    if len(messages) <= max_messages:
        return [SYSTEM_PROMPT] + [m for m in messages if not isinstance(m, SystemMessage)]
    
    recent = messages[-max_messages:]
    while recent and not isinstance(recent[0], (HumanMessage, SystemMessage)):
        recent.pop(0)
        
    return [SYSTEM_PROMPT] + [m for m in recent if not isinstance(m, SystemMessage)]

def chat_node(state: ChatState):
    messages = state["messages"]
    safe_history = get_safe_history(messages, max_messages=6)

    if safe_history and isinstance(safe_history[-1], ToolMessage):
        response = primary_model.invoke(safe_history)
    else:
        response = llm_with_tool.invoke(safe_history, stream=False)
        
    return {"messages": [response]}

db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=db_conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

chatbot = builder.compile(checkpointer=checkpointer)

def generate_title(first_message: str) -> str:
    prompt = [
        SystemMessage(content="Generate a 3-5 word concise title for this chat. Output ONLY title text."),
        HumanMessage(content=first_message)
    ]
    response = llm.invoke(prompt)
    return response.content.strip()