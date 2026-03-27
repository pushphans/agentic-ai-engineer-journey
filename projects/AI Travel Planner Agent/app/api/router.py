from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.travel_request import travel_request
from app.agent.travel_agent_graph import workflow, travel_state
from langchain.messages import AIMessage, HumanMessage


travel_router = APIRouter(prefix="/api/vi")


@travel_router.post("/plan-trip")
async def plan_trip(request: travel_request):
    async def generate_plan():
        try:
            config = {"configurable": {"thread_id": request.thread_id}}

            initial_state: travel_state = {
                "messages": [HumanMessage(content=request.message)]
            }
            async for chunk in workflow.astream(
                initial_state, config=config, stream_mode="messages"
            ):
                if chunk[0].content:
                    yield chunk[0].content

        except Exception as e:
            yield f"error : {e}"

    return StreamingResponse(generate_plan(), media_type="text/plain")
