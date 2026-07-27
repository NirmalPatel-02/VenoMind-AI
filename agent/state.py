from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class ChatState(TypedDict):
    query: Optional[str]
    answer: Optional[str]
    evaluation: Optional[Literal['approved', 'needs_improvement']]
    feedback: Optional[str]
    iteration: int
    max_iteration: int
    messages: Annotated[list[BaseMessage], add_messages]

class SearchInput(BaseModel):
    query: str = Field(description="Search terms for web query.")

class Answer_eval(BaseModel):
    evaluation: Literal['approved', 'needs_improvement'] = Field(
        ..., 
        description="Select 'approved' if the answer directly answers the query accurately and adheres to formatting. Select 'needs_improvement' if facts are missing, tone/formatting is off, or query is unanswered."
    )
    feedback: str = Field(
        ..., 
        description="Constructive, specific feedback on what is wrong and how to fix it."
    )