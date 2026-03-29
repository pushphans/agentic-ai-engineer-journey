from pydantic import BaseModel


class chat_request(BaseModel):
    message: str
    thread_id: str
