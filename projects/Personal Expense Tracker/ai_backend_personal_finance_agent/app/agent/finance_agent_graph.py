from email import message

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field
from app.core.config import settings
from langchain.messages import HumanMessage, SystemMessage
from app.core.finance_state import FinanceState
from datetime import datetime
from app.core.db_utils import (
    save_expenses,
    fetch_expenses,
    save_chat_messages,
    fetch_chat_history,
)

# llm
llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google-genai",
    api_key=settings.GOOGLE_API_KEY,
)


# structured output schema
class FinanceInfoSchema(BaseModel):
    amount: float = Field(description="Amount in Rupees for the given item")
    category: str = Field(
        description="Category of the given item. Eg: Food, Transport, Entertainment, etc."
    )
    item: str = Field(description="Name of the item. Eg: Pizza, Movie Ticket, etc.")


# structured llm
structured_llm = llm.with_structured_output(schema=FinanceInfoSchema)


# Nodes


async def info_extractor_node(state: FinanceState):
    last_message = state["messages"][-1].content
    user_id = state["user_id"]
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = SystemMessage(
        content=f"""
You are a helpful finance assistant. Extract the relevant information from the user's message and provide it in a structured format.
"""
    )

    response: FinanceInfoSchema = await structured_llm.ainvoke(
        [system_prompt] + [HumanMessage(content=last_message)]
    )

    amount = response.amount
    category = response.category
    item = response.item

    await save_expenses(
        user_id=user_id,
        amount=amount,
        category=category,
        item=item,
    )

    return {
        "extracted_info": {
            "amount": response.amount,
            "category": response.category,
            "item": response.item,
        }
    }


async def analysis_node(state: FinanceState) -> FinanceState:

    pass


# Graph
graph = StateGraph(FinanceState)

graph.add_node("info_extractor_node", info_extractor_node)
graph.add_node("analysis_node", analysis_node)


graph.add_edge(START, "info_extractor_node")
graph.add_edge("info_extractor_node", "analysis_node")
