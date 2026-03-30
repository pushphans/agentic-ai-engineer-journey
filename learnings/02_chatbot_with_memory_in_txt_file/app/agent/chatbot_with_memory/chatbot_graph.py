from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.core.agent_state import AgentState


# LLM INITIALIZATION
llm = init_chat_model(
    model="gpt-4o-mini", model_provider="groq", api_key=settings.GROQ_API_KEY
)


# NODES
async def chat_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


# GRAPH
graph = StateGraph(AgentState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
