import os 

import pytest
import torch 
from fastapi.testclient import TestClient

from app import create_app, get_model_path
from data import EOS_ID, SRC_VOCAB, TGT_VOCAB
from model import Transformer
from model_training import Checkpoint_path

@pytest.fixture
def client():
    # Example: untrained model + Stage 9 bias trick -> always says <eos> first -> words "".
    # Lets every test run without best.pt (CI, Docker build).
    torch.manual_seed(0)
    model = Transformer(src_vocab_size=len(SRC_VOCAB), tgt_vocab_size=len(TGT_VOCAB))
    with torch.no_grad():
        model.out.bias[EOS_ID] = 100.0
    app = create_app(model=model, epoch=0)
    with TestClient(app) as test_client:          # "with" runs the startup (lifespan)
        yield test_client


# ---------- /health ----------
def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checkpoint_epoch": 0}


# ---------- /translate, good orders ----------
def test_translate_returns_number_and_words(client):
    # Example: 4072, our usual card. Bias trick -> words "" (shape test, not a quality test)
    response = client.post("/translate", json={"number": 4072})
    assert response.status_code == 200
    assert response.json() == {"number": 4072, "words": ""}


@pytest.mark.parametrize("number", [0, 999_999])
def test_translate_accepts_edges(client, number):
    # Example: the smallest and largest cards data.py ever made
    response = client.post("/translate", json={"number": number})
    assert response.status_code == 200
    assert response.json()["number"] == number


# ---------- /translate, bad orders -> 422 ----------
@pytest.mark.parametrize("body, message_part", [
    ({"number": -5}, "greater than or equal to 0"),        # negative
    ({"number": 1_000_000}, "less than or equal to 999999"),  # above the biggest card
    ({"number": 40.72}, "valid integer"),                  # not a whole number
    ({"number": "4072"}, "valid integer"),                 # text, not a number (strict)
    ({}, "Field required"),                                # number missing
])
def test_translate_rejects_bad_input(client, body, message_part):
    response = client.post("/translate", json=body)
    assert response.status_code == 422
    assert message_part in response.json()["detail"][0]["msg"]


def test_reject_points_at_number_field(client):
    # the error says WHICH field is wrong: body -> number
    response = client.post("/translate", json={"number": -5})
    assert response.json()["detail"][0]["loc"] == ["body", "number"]


# ---------- Where the checkpoint lives ----------
def test_model_path_default(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    assert get_model_path() == Checkpoint_path


def test_model_path_from_env(monkeypatch):
    # Example: the path Docker will use in Stage 12
    monkeypatch.setenv("MODEL_PATH", "/app/model/best.pt")
    assert get_model_path() == "/app/model/best.pt"


def test_missing_checkpoint_fails_at_startup(monkeypatch, tmp_path):
    # no file there -> clear error at startup, not at the first request
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    app = create_app()
    with pytest.raises(RuntimeError, match="no checkpoint"):
        with TestClient(app):
            pass


# ---------- The real model (skips without best.pt) ----------
@pytest.mark.skipif(not os.path.exists(get_model_path()), reason="best.pt not here (CI / Docker)")
def test_real_checkpoint_4072():
    app = create_app()                            # loads the real best.pt once
    with TestClient(app) as test_client:
        response = test_client.post("/translate", json={"number": 4072})
        assert response.json() == {"number": 4072, "words": "four thousand seventy two"}
        assert test_client.get("/health").json()["checkpoint_epoch"] >= 1