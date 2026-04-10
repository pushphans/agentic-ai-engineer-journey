import asyncio

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from app.core.llm import (
    IntentSchema,
    TimeRangeSchema,
    llm,
    intent_llm,
    info_llm,
    time_llm,
)
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, Any, Literal, TypedDict
from app.core.db import fetch_expenses, save_expense
from datetime import datetime


# --------------------
# State Class
# --------------------
class AgentState(TypedDict):
    user_id: str
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["save-expense", "fetch-expense"]
    extracted_info: dict[str, Any]
    save_status: str
    fetch_status: str
    fetched_data: list[dict]


# --------------------
# Nodes
# --------------------
async def supervisor_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]

    system_message = SystemMessage(
        content=f"""
You are an intent classifier. You have to classify intent from user's query as either "save-expense" or "fetch_expense".
"""
    )

    response: IntentSchema = await intent_llm.ainvoke([system_message, user_message])

    return {"intent": response.intent}


async def intent_routing(state: AgentState) -> Literal["save-expense", "fetch-expense"]:
    intent = state["intent"]

    if intent == "save-expense":
        return "save-expense"
    elif intent == "fetch-expense":
        return "fetch-expense"
    else:
        raise ValueError(f"Unknown intent: {intent}")


async def extract_info_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]

    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = SystemMessage(
        content=f"""
You will extract values from user's message for amount they spent, category of expense, item they spent money on and date of the expense.
current date : {current_date}
"""
    )

    response = await info_llm.ainvoke([system_message, user_message])

    return {
        "extracted_info": {
            "amount": response.amount,
            "category": response.category,
            "item": response.item,
            "date": response.date,
        }
    }


async def save_to_db_node(state: AgentState) -> AgentState:
    extracted_info = state["extracted_info"]
    user_id = state["user_id"]
    session_id = state["session_id"]

    result = await save_expense(
        amount=extracted_info["amount"],
        category=extracted_info["category"],
        item=extracted_info["item"],
        date=extracted_info["date"],
        user_id=user_id,
        session_id=session_id,
    )

    if result == True:
        return {"save_status": "success"}

    else:
        return {"save_status": "error"}


async def extract_time_info_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_message = SystemMessage(
        content=f"""
You extract time ranges from user's query.
Current date: {current_date}

If user wants single range → query_type: "single", fill range1 only.
If user wants comparison → query_type: "comparison", fill both range1 and range2.

Examples:
- "last month ka kharcha" → single, range1: 2025-03-01 to 2025-03-31
- "last month vs this month" → comparison, range1: march, range2: april
"""
    )

    response: TimeRangeSchema = await time_llm.ainvoke([system_message, user_message])

    return {
        "extracted_info": {
            "query_type": response.query_type,
            "range1_start": response.range1_start,
            "range1_end": response.range1_end,
            "range2_start": response.range2_start,
            "range2_end": response.range2_end,
        }
    }


async def fetch_data_from_db_node(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    info = state["extracted_info"]

    # YE ADD KAR — dekho LLM ne kya nikala
    print(f"Query type: {info['query_type']}")
    print(f"Range1: {info['range1_start']} to {info['range1_end']}")
    print(f"Category filter: {info.get('category_filter')}")

    category_filter = info.get("category_filter")

    range1_data = await fetch_expenses(
        user_id=user_id,
        start_date=info["range1_start"],
        end_date=info["range1_end"],
        category_filter=category_filter,
    )

    range2_data = []

    if info["query_type"] == "comparison":
        range2_data = await fetch_expenses(
            user_id=user_id,
            start_date=info["range2_start"],
            end_date=info["range2_end"],
            category_filter=category_filter,
        )

    return {
        "fetch_status": "success" if range1_data else "empty",
        "fetched_data": {
            "query_type": info["query_type"],
            "range1": range1_data,
            "range2": range2_data,
        },
    }


async def advisor_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    messages = state["messages"]

    if intent == "save-expense":
        save_status = state["save_status"]
        extracted_info = state["extracted_info"]

        system_content = f"""
        You are a professional finance advisor.
        Don't make fun of user at any cost.
        User ne ek expense save kiya.
        Save status: {save_status}
        Item: {extracted_info['item']}, Amount: ₹{extracted_info['amount']}
        Funny way mein respond karo.
        """

    elif intent == "fetch-expense":
        fetched = state["fetched_data"]
        fetch_status = state["fetch_status"]
        info = state["extracted_info"]

        if fetch_status == "empty":
            summary = "No expenses found."

        else:
            range1 = fetched["range1"]
            range2 = fetched["range2"]

            total1 = sum(row["amount"] for row in range1)  # Python se calculate

            if fetched["query_type"] == "single":
                by_category = {}
                for row in range1:
                    cat = row["category"]
                    by_category[cat] = by_category.get(cat, 0) + row["amount"]

                summary = f"Total: ₹{total1}, By category: {by_category}"

            elif fetched["query_type"] == "comparison":
                total2 = sum(row["amount"] for row in range2)
                diff = total1 - total2

                summary = f"""
                Range 1: ₹{total1}
                Range 2: ₹{total2}
                Difference: ₹{abs(diff)} {'zyada' if diff > 0 else 'kam'} tha range 1 mein
                """

        system_content = f"""
        You are a professional finance advisor.
        Don't make fun of user at any cost.
        User ne expenses fetch karne ki request ki thi.
        Summary: {summary}
        Funny way mein respond karo.
        """

    system_message = SystemMessage(content=system_content)
    response = await llm.ainvoke([system_message] + messages)
    return {"messages": [response]}


# -------------------
# Graph
# -------------------
graph = StateGraph(AgentState)

graph.add_node("supervisor_node", supervisor_node)
graph.add_node("extract_info_node", extract_info_node)
graph.add_node("save_to_db_node", save_to_db_node)
graph.add_node("advisor_node", advisor_node)

graph.add_node("extract_time_info_node", extract_time_info_node)
graph.add_node("fetch_data_from_db_node", fetch_data_from_db_node)


graph.add_edge(START, "supervisor_node")
graph.add_conditional_edges(
    "supervisor_node",
    intent_routing,
    {
        "save-expense": "extract_info_node",
        "fetch-expense": "extract_time_info_node",
    },
)
graph.add_edge("extract_info_node", "save_to_db_node")
graph.add_edge("save_to_db_node", "advisor_node")
graph.add_edge("advisor_node", END)

graph.add_edge("extract_time_info_node", "fetch_data_from_db_node")
graph.add_edge("fetch_data_from_db_node", "advisor_node")

workflow = graph.compile()


# async def workflow_run():
#     initial_state: AgentState = {
#         "user_id": "0fb5a76c-c510-4712-ba59-9c02f5529022",
#         "session_id": "0fb5a76c-c510-4712-ba59-9c02f5529022",
#         "messages": [
#             HumanMessage(content="bro abhi tak total kitna kharcha ho gaya hai yrr")
#         ],
#         "intent": "",
#         "extracted_info": {},
#         "save_status": "",
#         "fetch_status": "",
#         "fetch_data": [],
#     }

#     final_state: AgentState = await workflow.ainvoke(initial_state)

#     print(final_state["messages"])


# asyncio.run(workflow_run())


# terminal output =>
# [HumanMessage(content='hey bro aaj maine 500 rupaye kharch diye dominos jakar pizza pe.', additional_kwargs={}, response_metadata={}, id='f7c10af2-cbb1-4f0f-a37d-e715fddcae3b'), HumanMessage(content="{'status': 'error', 'message': 'Something went wrong'}", additional_kwargs={}, response_metadata={}, id='3ffcb5d2-48f5-4d35-974e-af36a27434d3'), AIMessage(content='Oh ho! Lagta hai kuch tech ka jaadu kar raha hai, lekin chalo, jaise pizza ka slice do hissa hota hai, waise hum isko bhi do tukdo mein baant dete hain!\n\nToh, 500 rupees ka pizza kharcha karke, aapne na sirf apni pet ki khushi ki hai, balki aapne Dominos ke employees ko bhi ek khushi ka moka diya! Lekin ab samjo, pahuncha hua paisa maine tumhe bata diya hai, “Arre bhai, pizza humesha paise ka sahi istemaal nahi hota!” \n\nAb socho, agar aap yeh 500 rupees kisi savings account me daal dete, toh woh bhi thoda thoda karke pizza hi ban jate! Kyunki compounded interest bhi toh ek tarah se ‘extra cheese’ hai, samjhe na? 😄\n\nToh agli baar jab bhi tum pizza lo, thoda planning se le lo. Aur haan, ek slice mujhe bhi bejhna mat bhoolna, waise main slice se pehle hi financial advice de dunga! 🍕💸', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 247, 'prompt_tokens': 67, 'total_tokens': 314, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_262a1aea33', 'id': 'chatcmpl-DSi5XkHtofonYlFZTZLC6mGRvzdzk', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d720e-4421-7993-832b-101ccad9adbb-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 67, 'output_tokens': 247, 'total_tokens': 314, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
