from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages, AnyMessage
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from app.core.config import settings
import os
from langgraph.checkpoint.memory import InMemorySaver


os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY


# LLM
llm = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")


# OUTPUT STRUCTURE
class info_structure(BaseModel):
    destination: str = Field(description="The travel destination")
    budget: str = Field(description="The travel budget")
    days: int = Field(description="The number of days for the trip")


# STRUCTURED LLM
structured_llm = llm.with_structured_output(schema=info_structure)


# STATE CLASS
class travel_state(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    destination: Optional[str]
    budget: Optional[str]
    days: Optional[int]


# NODES
async def chat_node(state: travel_state) -> travel_state:
    messages = state["messages"]

    system_message = SystemMessage(
        content="You are a travel assistant. Ask the user for destination, budget, and number of days if not provided."
    )

    # system_message is a string and mesages is a list of string so system_message + messages will give flat list as LIST + LIST = LIST
    response: info_structure = await llm.ainvoke([system_message] + messages)
    return {"messages": [response]}


async def classifier_node(state: travel_state) -> travel_state:
    messages = state["messages"]
    system_message = SystemMessage(
        content="Extract destination, budget, days from conversation."
    )
    response: info_structure = await structured_llm.ainvoke([system_message] + messages)
    return {
        "destination": response.destination or state.get("destination"),
        "budget": response.budget or state.get("budget"),
        "days": response.days or state.get("days"),
    }


def routing_function(state: travel_state) -> Literal["chat_node", "plan_node"]:
    if state["destination"] and state["budget"] and state["days"]:
        return "plan_node"
    return "chat_node"


async def plan_node(state: travel_state) -> travel_state:
    destination = state["destination"]
    budget = state["budget"]
    days = state["days"]
    messages = state["messages"]

    system_message = SystemMessage(
        content=f"""
You are a travel assistant. Plan a trip to {destination} with a budget of {budget} for {days} days. Provide a detailed itinerary including activities, accommodations, and dining options.
"""
    )

    response = await llm.ainvoke([system_message] + messages)

    return {"messages": [response]}


graph = StateGraph(travel_state)

graph.add_node("chat_node", chat_node)
graph.add_node("classifier_node", classifier_node)
graph.add_node("plan_node", plan_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", "classifier_node")
graph.add_conditional_edges(
    "classifier_node",
    routing_function,
    {"chat_node": "chat_node", "plan_node": "plan_node"},
)
graph.add_edge("plan_node", END)

memory = InMemorySaver()

workflow = graph.compile(checkpointer=memory)
