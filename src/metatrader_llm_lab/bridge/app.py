from fastapi import FastAPI

from metatrader_llm_lab.ai.engine import decide
from metatrader_llm_lab.bridge.schemas import MarketSnapshot, TradeDecision

app = FastAPI(title="MetaTrader LLM Lab", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "lesson": "01", "ai": "not-connected"}


@app.post("/snapshot", response_model=TradeDecision)
def receive_snapshot(snapshot: MarketSnapshot) -> TradeDecision:
    return decide(snapshot)
