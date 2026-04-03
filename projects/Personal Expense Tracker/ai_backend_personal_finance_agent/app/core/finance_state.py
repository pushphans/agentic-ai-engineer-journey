from typing import Annotated, TypedDict, Optional
from langchain.messages import AnyMessage
from langgraph.graph import add_messages


class FinanceState(TypedDict):
    user_id: str
    messages: Annotated[list[AnyMessage], add_messages]

    extracted_info: Optional[dict]
