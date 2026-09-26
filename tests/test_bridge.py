from fastapi.testclient import TestClient

from metatrader_llm_lab.bridge.app import app
from metatrader_llm_lab.bridge.schemas import MarketSnapshot
from metatrader_llm_lab.features.market import build_market_features

client = TestClient(app)


def sample_snapshot() -> dict:
    candles = []
    for i in range(11):
        close = 100.0 + i
        candles.append(
            {
                "time": f"2026-09-22T{6 + i // 4:02d}:{(i % 4) * 15:02d}:00",
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "tick_volume": 1000 + i * 10,
            }
        )

    return {
        "symbol": "XAUUSD_l",
        "timeframe": "PERIOD_M15",
        "market": {"bid": 110.0, "ask": 110.42, "spread_points": 42},
        "candles": candles,
        "account": {"balance": 1000.0, "equity": 998.5, "free_margin": 950.2},
        "position": {
            "status": "NONE", "type": "NONE", "volume": 0.0,
            "open_price": 0.0, "profit": 0.0,
        },
        "timestamp": "2026-09-22T09:00:00",
    }


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["lesson"] == "02"


def test_snapshot_returns_wait_and_preserves_candle_sequence() -> None:
    snapshot = sample_snapshot()
    response = client.post("/snapshot", json=snapshot)
    assert response.status_code == 200
    assert response.json()["decision"] == "WAIT"

    latest = client.get("/snapshot/latest")
    assert latest.status_code == 200
    payload = latest.json()
    assert payload["symbol"] == "XAUUSD_l"
    assert len(payload["candles"]) == 11
    assert payload["candles"][-1]["close"] == 110.0
    assert payload["account"]["balance"] == 1000.0


def test_build_market_features() -> None:
    snapshot = MarketSnapshot.model_validate(sample_snapshot())
    features = build_market_features(snapshot)

    assert features.return_1 == (110.0 - 109.0) / 109.0
    assert features.return_3 == (110.0 - 107.0) / 107.0
    assert features.return_5 == (110.0 - 105.0) / 105.0
    assert features.return_10 == (110.0 - 100.0) / 100.0
    assert features.candle_range == 2.0
    assert features.candle_body == 0.5
    assert features.body_to_range == 0.25
