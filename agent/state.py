from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class SearchInput(BaseModel):
    query: str = Field(description="Search terms for web query.")
