"""
Smoke test (Stage 14): the REAL deployed service, over the internet.

Skipped unless SERVICE_URL is set, so plain `pytest` (laptop or CI) never touches the network.
From Git Bash:
    export SERVICE_URL=https://number-words-....run.app
    python -m pytest tests/test_live.py -v
"""

import json
import os
import urllib.request

import pytest

Service_url = os.environ.get("SERVICE_URL", "").rstrip("/")      # "https://.../" -> "https://..."
Timeout = 60                                                      # seconds; min 0 -> first call may cold-start

pytestmark = pytest.mark.skipif(not Service_url, reason="SERVICE_URL not set (live smoke test only)")


# ---------- Helpers ----------
def get_json(path):
    # Example: get_json(path="/health") -> {"status": "ok", "checkpoint_epoch": 18}
    request = urllib.request.Request(url=Service_url + path, method="GET")
    with urllib.request.urlopen(request, timeout=Timeout) as response:
        assert response.status == 200
        return json.loads(response.read())


def post_json(path, body):
    # Example: post_json(path="/translate", body={"number": 4072}) -> {"number": 4072, "words": "..."}
    request = urllib.request.Request(url=Service_url + path,
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=Timeout) as response:
        assert response.status == 200
        return json.loads(response.read())


# ---------- The live service ----------
def test_live_health_reports_epoch_18():
    assert get_json(path="/health") == {"status": "ok", "checkpoint_epoch": 18}


def test_live_translates_4072():
    answer = post_json(path="/translate", body={"number": 4072})
    assert answer == {"number": 4072, "words": "four thousand seventy two"}