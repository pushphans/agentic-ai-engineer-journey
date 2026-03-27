from fastapi import FastAPI
from app.api.router import travel_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ok"}


app.include_router(router=travel_router)