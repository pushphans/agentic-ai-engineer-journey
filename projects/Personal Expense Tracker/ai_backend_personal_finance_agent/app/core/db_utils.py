from langchain.messages import AIMessage, HumanMessage
from supabase import create_async_client, AsyncClient
import supabase

from app.core.config import settings
from datetime import datetime, timedelta, timezone


async def get_supabase_client() -> AsyncClient:
    return await create_async_client(
        supabase_key=settings.SUPABASE_KEY, supabase_url=settings.SUPABASE_URL
    )


async def save_chat_messages(
    user_id: str, session_id: str, user_text: str, ai_text: str
):

    try:

        supabase = await get_supabase_client()
        now = datetime.now(timezone.utc)

        data_to_insert = [
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": "user",
                "content": user_text,
                "created_at": now.isoformat(),
            },
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": "ai",
                "content": ai_text,
                "created_at": (now + timedelta(seconds=1)).isoformat(),
            },
        ]

        await supabase.table("finance_agent_chat_history").insert(
            data_to_insert
        ).execute()
        print("Chat messages saved successfully.")
    except Exception as e:
        print(f"❌ Error saving history: {e}")


async def fetch_chat_history(user_id: str, session_id: str, limit: int = 10):
    try:
        supabase = await get_supabase_client()

        response = (
            await supabase.table("finance_agent_chat_history")
            .select("*")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        db_rows = response.data

        chat_history = []

        for row in db_rows:
            if row["role"] == "user":
                chat_history.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                chat_history.append(AIMessage(content=row["content"]))

        chat_history.reverse()  # To get the messages in chronological order

        return chat_history

    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []


async def save_expenses(user_id: str, item: str, amount: float, category: str):

    try:
        supabase = await get_supabase_client()

        data_to_insert = {
            "user_id": user_id,
            "item": item,
            "amount": amount,
            "category": category,
        }

        supabase.table("finance_agent_expenses").insert(data_to_insert).execute()
        print("Expenses saved successfully.")

    except Exception as e:
        print(f"❌ Error saving expenses: {e}")


async def fetch_expenses(user_id: str, days: int):

    try:

        supabase = await get_supabase_client()
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        response = (
            supabase.table("finance_agent_expenses")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", start_date)
            .execute()
        )
        db_rows = response.data

        # Python mein saare amounts ko plus kar lo
        total_spent = sum(float(exp["amount"]) for exp in response)

        category_summary = {}
        for exp in response:
            cat = exp["category"]
            amt = float(exp["amount"])
            category_summary[cat] = category_summary.get(cat, 0) + amt

        return {
            "total_spent": total_spent,
            "category_summary": category_summary,
            "period": f"Last {days} days",
        }

    except Exception as e:
        print(f"❌ Error fetching expenses: {e}")
        return {"total_spent": 0, "category_summary": {}}
