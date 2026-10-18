"""
Serve (Stage 11): the trained model behind a web API

POST /translate  {"number": 4072} -> {"number": 4072, "words": "four thousand seventy two"}
GET  /health     -> {"status": "ok", "checkpoint_epoch": 18}

best.pt is loaded ONCE at startup (lifespan), kept on app.state, borrowed by every request.
Tests hand in their own model with create_app(model=..., epoch=...), so they never need best.pt.
Run with:  python -m uvicorn app:app --port 8000     then open http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, StrictInt

from model_training import Checkpoint_path, load_checkpoint
from translate import greedy_translate

Device = "cpu"                                        # one card is tiny; Cloud Run has no GPU by default
Max_number = 999_999                                  # data.py only ever made cards 0 .. 999,999


# ---------- What a valid order and answer look like ----------
class TranslateRequest(BaseModel):
    # StrictInt: 4072 ok; 40.72, "4072", true -> 422. ge/le = greater/less than or equal
    number: StrictInt = Field(ge=0, le=Max_number, examples=[4072])


class TranslateResponse(BaseModel):
    number: int                                       # echoed back: 4072
    words: str                                        # "four thousand seventy two"


class HealthResponse(BaseModel):
    status: str                                       # "ok"
    checkpoint_epoch: int                             # 18 for our best.pt


# ---------- Where the checkpoint lives ----------
def get_model_path():
    # MODEL_PATH set from outside (Docker / Cloud Run later), else checkpoints/best.pt
    return os.environ.get("MODEL_PATH", Checkpoint_path)


# ---------- Build the app ----------
def create_app(model=None, epoch=None):
    # model=None -> real run, load best.pt at startup; model given -> tests, skip loading

    @asynccontextmanager
    async def lifespan(fastapi_app):
        # code before yield = startup (runs ONCE), code after yield = shutdown
        if model is None:
            path = get_model_path()
            if not os.path.exists(path):
                raise RuntimeError(f"no checkpoint at {path}: train first, or set MODEL_PATH")
            loaded_model, checkpoint = load_checkpoint(path=path, device=Device)
            fastapi_app.state.model = loaded_model
            fastapi_app.state.epoch = checkpoint["epoch"]
        else:
            fastapi_app.state.model = model.to(Device).eval()                 # dropout off
            fastapi_app.state.epoch = epoch if epoch is not None else 0
        yield

    fastapi_app = FastAPI(title="Number to words Transformer", lifespan=lifespan)

    @fastapi_app.post("/translate", response_model=TranslateResponse)
    def translate_number(body: TranslateRequest, request: Request):
        # body already passed the checks; plain def -> FastAPI runs it in a helper thread
        words = greedy_translate(model=request.app.state.model, number=body.number, device=Device)
        return TranslateResponse(number=body.number, words=words)

    @fastapi_app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        # never runs the model -> instant
        return HealthResponse(status="ok", checkpoint_epoch=request.app.state.epoch)

    return fastapi_app


app = create_app()                                    # what uvicorn runs: "app:app" = file app.py, variable app