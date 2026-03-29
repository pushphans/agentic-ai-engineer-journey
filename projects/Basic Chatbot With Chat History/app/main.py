from fastapi import FastAPI
from app.api.chat_router import chat_router

app = FastAPI()

app.include_router(chat_router)


@app.get("/")
def root():
    return {"status": "ok"}
