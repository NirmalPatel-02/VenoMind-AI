import sqlite3
import json
from pydantic import BaseModel, Field
from ddgs import DDGS
import yfinance as yf
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from config.settings import llm, DB_PATH, primary_model, title_primary, title_fallback
from agent.state import ChatState, SearchInput, StockInput, Answer_eval
from typing import Literal

SYSTEM_PROMPT = SystemMessage(content="""You are VenoMind AI, an expert AI assistant.
Always format your responses using rich Markdown:
- Use code blocks with language identifiers for code.
- Use Markdown tables (| Header | Header |) for structured data or metrics.
- Use bold text and bullet points for readability.

STRICT REAL-TIME DATA RULE:
1. Whenever the user asks for stock prices, use the `get_stock_price` tool — never guess a number.
2. Whenever the user asks for news, headlines, or "the latest" on a topic, use `web_news_search` — it returns real dated headlines, not a generic search summary.
3. For everything else that needs current information, use `web_search`.
4. DO NOT guess, fabricate, or predict future stock prices out of memory. Base all numbers ONLY on tool output.
5. If a tool returns no useful data, say so explicitly instead of inventing an answer.
6. Do NOT call any tool for stable, well-known knowledge that doesn't change day to day — answer directly from what you know.
7. Never describe your tools by name to the user. Just answer naturally.

CRITICAL TOOL CALL FORMAT:
When calling a function/tool, ALWAYS use strictly valid, double-quoted JSON arguments e.g. {"query": "Gujarat news"}. Never use parentheses or single quotes.
""")

EVALUATOR_PROMPT = """You are a strict Quality Assurance Evaluator for VenoMind AI.
Your job is to critically evaluate whether the generated AI Answer completely satisfies the User Query given the chat context.

Evaluation Checklist:
1. Accuracy & Completeness: Did the answer address ALL parts of the user query?
2. Real-time / Tool Usage: If real-time or factual data was required, did it use tool outputs accurately?
3. Clarity & Structure: Is the answer well-structured and easy to follow?
4. Conciseness: Is it free of repetitive fluff?

Rules for Decision:
- Mark 'approved' ONLY if the answer is accurate, complete, and helpful.
- Do NOT mark 'needs_improvement' solely for minor formatting preferences.
- If marking 'needs_improvement', set improvement_type:
  - 'needs_more_data' — missing/wrong real-time information.
  - 'needs_rewrite' — factual content is fine, but structure/clarity needs polishing.
- Provide actionable, specific feedback.
"""

OPTIMIZER_SYSTEM_PROMPT = SystemMessage(content="""You are VenoMind AI's Master Refinement Engine.
Your sole job is to rewrite a draft response based on critique feedback.

CRITICAL OUTPUT RULES:
1. Fix the problem described in the feedback.
2. Output ONLY the final, polished answer meant for the end-user.
3. DO NOT include meta-headers ("Improved Response:", "Version 2:").
4. DO NOT include disclaimers or changelogs at the bottom.
5. Maintain expert Markdown formatting.
6. Use ONLY facts/numbers/dates from the retrieved source data.
""")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Searches the web for general/current information. Do NOT use for stock prices or news headlines."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return json.dumps({"results": [], "note": "No results found."})
        cleaned = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:350]}
            for r in results
        ]
        return json.dumps({"results": cleaned}, ensure_ascii=False)
    except Exception as e:
        return f"Error executing search: {str(e)}"

@tool("web_news_search", args_schema=SearchInput)
def web_news_search(query: str) -> str:
    """Searches for recent NEWS specifically. Use whenever user asks for 'news', 'headlines', or 'the latest'."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
        if not results:
            return json.dumps({"results": [], "note": "No recent news found."})
        cleaned = []
        seen = set()
        for r in results:
            title = r.get("title", "")
            if title in seen:
                continue
            seen.add(title)
            cleaned.append({
                "title": title,
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", "")[:300],
            })
        return json.dumps({"results": cleaned}, ensure_ascii=False)
    except Exception as e:
        return f"Error executing news search: {str(e)}"

@tool("get_stock_price", args_schema=StockInput)
def get_stock_price(ticker: str) -> str:
    """Fetches real current price from Yahoo Finance."""
    ticker = ticker.strip().upper()
    candidates = [ticker]
    if not any(ticker.endswith(sfx) for sfx in (".NS", ".BO", ".L", ".TO")):
        candidates += [f"{ticker}.NS", f"{ticker}.BO"]

    for candidate in candidates:
        try:
            info = dict(yf.Ticker(candidate).fast_info)
            last_price = info.get("lastPrice")
            if last_price:
                r = lambda v: round(v, 2) if isinstance(v, (int, float)) else v
                return json.dumps({
                    "ticker": candidate,
                    "last_price": r(last_price),
                    "day_high": r(info.get("dayHigh")),
                    "day_low": r(info.get("dayLow")),
                    "previous_close": r(info.get("previousClose")),
                    "currency": info.get("currency"),
                })
        except Exception:
            continue

    return json.dumps({"error": f"No price data found for '{ticker}'."})

tools = [web_search, web_news_search, get_stock_price]
llm_with_tool = primary_model.bind_tools(tools, tool_choice="auto")

def get_safe_history(messages: list[BaseMessage], max_messages=6) -> list[BaseMessage]:
    """Retrieves clean conversation history while filtering out stale ToolMessages."""
    clean_history = []
    for m in messages:
        if isinstance(m, (HumanMessage, SystemMessage)):
            clean_history.append(m)
        elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            clean_history.append(m)

    recent = clean_history[-max_messages:] if len(clean_history) > max_messages else clean_history
    return [SYSTEM_PROMPT] + [m for m in recent if not isinstance(m, SystemMessage)]

MAX_TOOL_ROUNDS = 3 

def chat_node(state: ChatState):
    """Generates a response or tool call with fallback error protection."""
    messages = state["messages"]
    user_query = state.get("query")
    
    if not user_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

    safe_history = get_safe_history(messages, max_messages=6)

    # Check for tool messages generated ONLY in current turn
    last_human_idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    current_turn_messages = messages[last_human_idx:]
    current_tool_messages = [m for m in current_turn_messages if isinstance(m, ToolMessage)]

    if current_tool_messages:
        raw_context = "\n\n".join([f"Source Output:\n{m.content}" for m in current_tool_messages])
        synthesis_prompt = [
            SYSTEM_PROMPT,
            HumanMessage(content=(
                f"User Question: {user_query}\n\n"
                f"Fresh Data Retrieved For THIS Question:\n{raw_context}\n\n"
                "Synthesize the retrieved source data above into a concise, accurate Markdown answer. "
                "Base your answer ONLY on the retrieved data above."
            ))
        ]
        try:
            response = primary_model.invoke(synthesis_prompt)
        except Exception:
            response = AIMessage(content=f"Here is what I retrieved:\n\n{raw_context}")
    else:
        # Try invoking tool binding; if Groq API throws tool_use_failed, fallback to direct text invocation
        try:
            response = llm_with_tool.invoke(safe_history)
        except Exception:
            fallback_prompt = safe_history + [HumanMessage(content="Please answer the question directly in plain text.")]
            response = primary_model.invoke(fallback_prompt)

    final_text = str(response.content) if response.content else ""

    return {
        "messages": [response],
        "query": user_query or "General Query",
        "answer": final_text,
        "max_iteration": state.get("max_iteration", 2),
        "evaluation": state.get("evaluation"),
        "improvement_type": state.get("improvement_type"),
    }

evaluation_llm = title_primary.with_structured_output(Answer_eval).with_fallbacks(
    [title_fallback.with_structured_output(Answer_eval)]
)

def evaluate_answer(state: ChatState):
    messages = state["messages"]
    safe_history = get_safe_history(messages, max_messages=6)
    question = state.get('query', '')
    answer = state.get('answer', '')

    eval_prompt = [
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=f"""
            Conversation Context: {safe_history}
            User Question: {question}
            AI Answer Generated: {answer}
        """)
    ]

    current_iteration = state.get("iteration", 0)

    try:
        res: Answer_eval = evaluation_llm.invoke(eval_prompt)
    except Exception as e:
        return {
            'evaluation': 'approved',
            'improvement_type': None,
            'feedback': f'(Evaluator step skipped: {e})',
            'iteration': current_iteration,
        }

    if res.evaluation == "needs_improvement":
        current_iteration += 1

    return {
        'evaluation': res.evaluation,
        'improvement_type': res.improvement_type,
        'feedback': res.feedback,
        'iteration': current_iteration,
    }

def optimize_answer(state: ChatState):
    answer = state.get('answer', '')
    feedback = state.get('feedback', '')

    tool_outputs = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    source_block = (
        f"\n\nRaw source data retrieved earlier:\n{tool_outputs}"
        if tool_outputs else ""
    )

    opt_prompt = [
        OPTIMIZER_SYSTEM_PROMPT,
        HumanMessage(content=f"""
            Previous Draft:
            {answer}

            Critique & Feedback for Improvement:
            {feedback}
            {source_block}

            Rewrite the answer incorporating the feedback above into clean Markdown.
        """)
    ]

    response = primary_model.invoke(opt_prompt)

    return {
        "messages": [response],
        "answer": response.content,
    }

def route_after_chat(state: ChatState) -> Literal["tools", "evaluate_answer"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "evaluate_answer"

def route_after_evaluation(state: ChatState) -> Literal["optimize_answer", "chat_node", "__end__"]:
    eval_status = state.get("evaluation")
    iteration = state.get("iteration", 0)
    max_iteration = state.get("max_iteration", 2)

    if eval_status == "needs_improvement" and iteration < max_iteration:
        if state.get("improvement_type") == "needs_more_data":
            return "chat_node"
        return "optimize_answer"
    return END

db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=db_conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("evaluate_answer", evaluate_answer)
builder.add_node("optimize_answer", optimize_answer)

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", route_after_chat, {"tools": "tools", "evaluate_answer": "evaluate_answer"})
builder.add_edge("tools", "chat_node")
builder.add_conditional_edges(
    "evaluate_answer",
    route_after_evaluation,
    {"optimize_answer": "optimize_answer", "chat_node": "chat_node", END: END},
)
builder.add_edge("optimize_answer", "evaluate_answer")

chatbot = builder.compile(checkpointer=checkpointer)

def generate_title(first_message: str) -> str:
    prompt = [
        SystemMessage(content="Generate a 3-5 word concise title for this chat. Output ONLY title text."),
        HumanMessage(content=first_message)
    ]
    response = llm.invoke(prompt)
    return response.content.strip()