from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


# MODELS
llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")


# STATES
class chatbot_states(TypedDict):
    messages: Annotated[List[str], add_messages]


# NODE
def chat_node(state: chatbot_states) -> chatbot_states:
    messages = state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}


# GRAPH
graph = StateGraph(chatbot_states)
graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# MEMORY CHECKPOINTER
memory = MemorySaver()


workflow = graph.compile(checkpointer=memory)


initial_state: chatbot_states = {
    "messages": [HumanMessage(content="give me an essay on ai")]
}


# initial_state_2: chatbot_states = {
#     "messages": [HumanMessage(content="what is my name bro?")]
# }

# initial_state_3: chatbot_states = {
#     "messages": [HumanMessage(content="what did i asked you till now?")]
# }

thread_id = "user_123"

# CONFIG USING THREAD ID
config = {"configurable" : {"thread_id": thread_id}}

# final_state: chatbot_states = workflow.invoke(initial_state_2, config=config)
# final_state: chatbot_states = workflow.invoke(initial_state_3, config=config)

# final_state: chatbot_states = workflow.invoke(initial_state, config=config)


for chunks in workflow.stream(initial_state, config=config, stream_mode="messages"):
    if chunks[0].content:
        print(chunks[0].content, end="", flush=True)

print()
# print(final_state["messages"])

# print(final_state["messages"][-1].content)
