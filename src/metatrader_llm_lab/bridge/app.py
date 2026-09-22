from fastapi import FastAPI, HTTPException

from metatrader_llm_lab.ai.engine import decide
from metatrader_llm_lab.bridge.schemas import MarketSnapshot, TradeDecision

app = FastAPI(title="MetaTrader LLM Lab", version="0.1.1")

_latest_snapshot: MarketSnapshot | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "lesson": "01", "ai": "not-connected"}


@app.get("/snapshot/latest", response_model=MarketSnapshot)
def latest_snapshot() -> MarketSnapshot:
    """Return the most recent market snapshot received by the bridge."""
    if _latest_snapshot is None:
        raise HTTPException(status_code=404, detail="No snapshot received yet")
    return _latest_snapshot


@app.post("/snapshot", response_model=TradeDecision)
def receive_snapshot(snapshot: MarketSnapshot) -> TradeDecision:
    global _latest_snapshot
    _latest_snapshot = snapshot
    return decide(snapshot)
