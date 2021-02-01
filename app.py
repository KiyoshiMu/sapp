import json
import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sapp.search import Server

server = None
app = FastAPI()
index_p = "index_s.bin"
embed_p = "embed_s.pkl"
case_p = "cases.json"
path_p = "idx_path.json"


class Item(BaseModel):
    text: str


origins = json.loads(
    os.getenv("origins", '["http://127.0.0.1", "https://kkkfff.web.app"]')
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
        server = Server(index_p=index_p, embed_p=embed_p, case_p=case_p, path_p=path_p)
    return server.search(item.text, top=top)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
