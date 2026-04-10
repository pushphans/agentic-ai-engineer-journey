from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from langchain.messages import HumanMessage, AIMessage

from app.agent.finance_agent_graph import workflow, AgentState
from app.core.db import get_supabase_client

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    user_message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ExpenseResponse(BaseModel):
    id: str
    amount: float
    category: str
    item: str
    date: str


class ExpensesListResponse(BaseModel):
    expenses: list[ExpenseResponse]
    total: float
    count: int


async def fetch_chat_history(supabase, session_id: str, limit: int = 10) -> list:
    """Fetch recent chat history from Supabase for the given session."""
    try:
        result = (
            await supabase.table("finance_agent_chat_history")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        # Reverse to get chronological order (oldest first)
        messages = result.data[::-1] if result.data else []

        # Convert to LangChain message format
        history = []
        for msg in messages:
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history.append(AIMessage(content=msg["content"]))

        return history

    except Exception as e:
        print(f"Error fetching chat history: {str(e)}")
        return []


async def save_chat_message(supabase, session_id: str, user_id: str, role: str, content: str):
    """Save a chat message to Supabase."""
    try:
        await supabase.table("finance_agent_chat_history").insert({
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        print(f"Error saving chat message: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint that processes user messages through the finance agent.
    Fetches chat history from Supabase and saves new messages.
    """
    try:
        supabase = await get_supabase_client()

        # Fetch chat history from Supabase
        history = await fetch_chat_history(supabase, request.session_id)

        # Build messages list with history + current user message
        messages = history + [HumanMessage(content=request.user_message)]

        # Build initial state for the agent
        initial_state: AgentState = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "messages": messages,
            "intent": "",
            "extracted_info": {},
            "save_status": "",
            "fetch_status": "",
            "fetched_data": [],
        }

        # Run the agent workflow
        final_state = await workflow.ainvoke(initial_state)

        # Get the AI response from the last message
        ai_response = final_state["messages"][-1]
        response_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)

        # Save user message to history
        await save_chat_message(
            supabase, request.session_id, request.user_id, "user", request.user_message
        )

        # Save AI response to history
        await save_chat_message(
            supabase, request.session_id, request.user_id, "ai", response_text
        )

        return ChatResponse(response=response_text, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.get("/expenses", response_model=ExpensesListResponse)
async def get_expenses(
    user_id: str = Query(..., description="User ID to fetch expenses for"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to fetch"),
):
    """
    Fetch expenses from PostgreSQL with optional filters.
    Returns list of expenses along with total amount and count.
    """
    try:
        supabase = await get_supabase_client()

        # Build query
        query = (
            supabase.table("finance_agent_expense")
            .select("id, amount, category, item, date_of_expense")
            .eq("user_id", user_id)
            .limit(limit)
        )

        # Apply date filters
        if start_date:
            query = query.gte("date_of_expense", start_date + "T00:00:00+00:00")
        if end_date:
            query = query.lte("date_of_expense", end_date + "T23:59:59+00:00")

        # Apply category filter
        if category:
            query = query.eq("category", category)

        # Order by date descending (newest first)
        query = query.order("date_of_expense", desc=True)

        result = await query.execute()
        expenses_data = result.data or []

        # Transform response
        expenses = [
            ExpenseResponse(
                id=row["id"],
                amount=row["amount"],
                category=row["category"],
                item=row["item"],
                date=row["date_of_expense"].split("T")[0],  # Extract date part
            )
            for row in expenses_data
        ]

        total = sum(exp.amount for exp in expenses)

        return ExpensesListResponse(
            expenses=expenses,
            total=total,
            count=len(expenses),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching expenses: {str(e)}")
