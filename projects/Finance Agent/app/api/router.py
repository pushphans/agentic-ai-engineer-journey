from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.messages import HumanMessage, AIMessage
from app.agent.finance_agent_graph import workflow, AgentState
from app.core.db import get_supabase_client
from datetime import datetime

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    user_message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


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
