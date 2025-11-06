from fastapi import FastAPI

# when deploying as an independent service,
# copy this to the root of the src and change the import path
# from tools.generation.stem_diff import generate_stem_diff
from stem_diff import generate_stem_diff
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


class StemDiffInput(BaseModel):
    context_song_info: Dict[str, Any]
    prompt_stem_info: List[Dict[str, Any]]
    mix_stem_diff: List[Dict[str, Any]]
    task_id: str
    env: Optional[str] = "dev"


@app.post("/request_stem_diff")
def request_stem_diff(request: StemDiffInput):
    generated_stem_diff = asyncio.run(
        generate_stem_diff(
            context_song_info=request.context_song_info,
            prompt_stem_info=request.prompt_stem_info,
            mix_stem_diff=request.mix_stem_diff,
            task_id=request.task_id,
            env=request.env,
        )
    )
    return generated_stem_diff


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# for production
# gunicorn app:app -w 16 -k uvicorn.workers.UvicornWorker
