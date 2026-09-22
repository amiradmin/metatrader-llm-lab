from metatrader_llm_lab.bridge.schemas import MarketSnapshot, TradeDecision


def decide(snapshot: MarketSnapshot) -> TradeDecision:
    """Return a safe placeholder decision until a real model is connected."""
    _ = snapshot
    return TradeDecision(
        decision="WAIT",
        confidence=0.0,
        reason="AI model is not connected yet",
    )
