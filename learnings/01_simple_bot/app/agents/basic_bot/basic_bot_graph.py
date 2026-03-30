from langgraph.graph import StateGraph, START, END
from app.core.agent_state import AgentState
from langchain.chat_models import init_chat_model
from app.core.config import settings

# LLM INITIALIZATION
llm = init_chat_model(
    model="openai/gpt-oss-120b", model_provider="groq", api_key=settings.GROQ_API_KEY
)


# NODE FUNCTIONS
async def chat_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    response = await llm.ainvoke(messages)

    return {"messages": [response]}


# GRAPH CONSTRUCTION
graph = StateGraph(AgentState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# GRAPH COMPILATION
app = graph.compile()
