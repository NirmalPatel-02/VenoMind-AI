from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class ChatState(TypedDict):
    query: Optional[str]
    answer: Optional[str]
    evaluation: Optional[Literal['approved', 'needs_improvement']]
    improvement_type: Optional[Literal['needs_more_data', 'needs_rewrite']]
    feedback: Optional[str]
    iteration: int
    max_iteration: int
    messages: Annotated[list[BaseMessage], add_messages]

class SearchInput(BaseModel):
    query: str = Field(description="Search terms for web query.")

class StockInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, TSLA, RELIANCE.NS")

class Answer_eval(BaseModel):
    evaluation: Literal['approved', 'needs_improvement'] = Field(
        ...,
        description="Select 'approved' if the answer directly answers the query accurately. Select 'needs_improvement' if facts are missing/wrong or the query is unanswered."
    )
    improvement_type: Optional[Literal['needs_more_data', 'needs_rewrite']] = Field(
        default=None,
        description=(
            "Only set when evaluation is 'needs_improvement'. "
            "'needs_more_data' if the answer is missing, wrong, or has insufficient real-time/factual "
            "information and needs an actual re-search. "
            "'needs_rewrite' if the underlying facts are fine but clarity, completeness, or structure "
            "needs polishing — no new search required."
        )
    )
    feedback: str = Field(
        ...,
        description="Constructive, specific feedback on what is wrong and how to fix it."
    )