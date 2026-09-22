from fastapi.testclient import TestClient

from metatrader_llm_lab.bridge.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_snapshot_returns_wait() -> None:
    response = client.post(
        "/snapshot",
        json={
            "symbol": "XAUUSD_l",
            "timeframe": "M15",
            "bid": 4353.67,
            "ask": 4354.09,
            "spread_points": 42,
            "position": "NONE",
            "timestamp": "2026-09-22T07:15:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "WAIT"
    assert payload["confidence"] == 0.0
