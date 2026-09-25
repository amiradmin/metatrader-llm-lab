from fastapi.testclient import TestClient

from metatrader_llm_lab.bridge.app import app

client = TestClient(app)


def sample_snapshot() -> dict:
    return {
        "symbol": "XAUUSD_l",
        "timeframe": "PERIOD_M15",
        "market": {"bid": 4353.67, "ask": 4354.09, "spread_points": 42},
        "last_closed_candle": {
            "open": 4358.10, "high": 4359.40, "low": 4352.80,
            "close": 4353.67, "tick_volume": 1842,
        },
        "account": {"balance": 1000.0, "equity": 998.5, "free_margin": 950.2},
        "position": {
            "status": "NONE", "type": "NONE", "volume": 0.0,
            "open_price": 0.0, "profit": 0.0,
        },
        "timestamp": "2026-09-22T07:15:00",
    }


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["lesson"] == "02"


def test_snapshot_returns_wait_and_is_available_as_latest() -> None:
    snapshot = sample_snapshot()
    response = client.post("/snapshot", json=snapshot)
    assert response.status_code == 200
    assert response.json()["decision"] == "WAIT"

    latest = client.get("/snapshot/latest")
    assert latest.status_code == 200
    payload = latest.json()
    assert payload["symbol"] == "XAUUSD_l"
    assert payload["last_closed_candle"]["tick_volume"] == 1842
    assert payload["account"]["balance"] == 1000.0
