from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from app.core.config import settings
from datetime import datetime


llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    api_key=settings.OPENAI_API_KEY,
)


class IntentSchema(BaseModel):
    intent: Literal["save-expense", "fetch-expense"] = Field(
        description="Intent of the user's query."
    )


intent_llm = llm.with_structured_output(schema=IntentSchema)


class InfoExtractionSchema(BaseModel):
    amount: float = Field(description="Total amount spent by user.")
    category: str = Field(description="Category of the expense")
    item: str = Field(description="Item name")
    date: datetime = Field(description="Date of the expense")


info_llm = llm.with_structured_output(schema=InfoExtractionSchema)


class TimeRangeSchema(BaseModel):
    query_type: Literal["single", "comparison"] = Field(
        description="Type of the query from user. If user wants his expenses for a single range like today, yesterday, this month, last two months etc use 'single' and if user want comparison like today vs tommorow, this month vs last month etc use 'comparison'"
    )
    range1_start: str = Field(description="Start date of the range")  # YYYY-MM-DD
    range1_end: str = Field(description="End date of the range")  # YYYY-MM-DD
    range2_start: str | None = Field(
        default=None,
        description="Start date of range 2 in YYYY-MM-DD format. Only for comparison, else None.",
    )
    range2_end: str | None = Field(
        default=None,
        description="End date of range 2 in YYYY-MM-DD format. Only for comparison, else None.",
    )
    category_filter: str | None = Field(
        default=None,
        description="Category to filter by. Example: 'Food', 'Travel'. None if user wants all categories.",
    )


time_llm = llm.with_structured_output(schema=TimeRangeSchema)
