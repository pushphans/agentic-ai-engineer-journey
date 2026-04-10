from datetime import datetime

from supabase import AsyncClient, acreate_client
from app.core.config import settings


async def get_supabase_client() -> AsyncClient:
    return await acreate_client(
        supabase_url=settings.SUPABASE_URL, supabase_key=settings.SUPABASE_KEY
    )


async def save_expense(
    user_id: str,
    session_id: str,
    amount: float,
    category: str,
    item: str,
    date: datetime,
) -> bool:
    try:
        supabase = await get_supabase_client()

        rows = {
            "user_id": user_id,
            "session_id": session_id,
            "amount": amount,
            "category": category,
            "item": item,
            "date_of_expense": date.isoformat(),
        }

        result = await supabase.table("finance_agent_expense").insert(rows).execute()

        print(f"✅ Save successful: {result}")
        print("\n")

        return True

    except Exception as e:
        print(f"Error in saving data in db {str(e)}")
        print("\n")
        return False


async def fetch_expenses(
    user_id: str,
    start_date: str,
    end_date: str,
    category_filter: str | None = None,
) -> list[dict]:

    try:

        supabase = await get_supabase_client()

        query = (
            supabase.table("finance_agent_expense")
            .select("amount, item, category, date_of_expense")
            .eq("user_id", user_id)
            .gte("date_of_expense", start_date + "T00:00:00+00:00")
            .lte("date_of_expense", end_date + "T23:59:59+00:00")
        )

        if category_filter:
            query = query.eq("category", category_filter)

        result = await query.execute()
        print(f"✅ Fetched successful: {result}")
        print("\n")

        return result.data

    except Exception as e:
        print(f"Error in fetching data from db {str(e)}")
        return []
