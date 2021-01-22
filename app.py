import json
import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sapp.search import Server

server = None
app = FastAPI()


class Item(BaseModel):
    text: str


origins = json.loads(
    os.getenv("origins", '["http://localhost:8080", "https://kkkfff.web.app"]')
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search/")
async def search(
    item: Item,
    top: int = Query(
        default=10,
        gt=1,
        lt=100,
        title="top",
        description="The number of needed samples. 1 < number < 100",
    ),
):
    global server
    if not server:
        server = Server()
    return server.search(item.text, top=top)
