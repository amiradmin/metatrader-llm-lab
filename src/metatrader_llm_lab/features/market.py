from dataclasses import dataclass

from metatrader_llm_lab.bridge.schemas import MarketSnapshot


@dataclass(frozen=True)
class MarketFeatures:
    """Small, explainable numerical features derived from closed candles."""

    return_1: float
    candle_range: float
    candle_body: float
    body_to_range: float


def build_market_features(snapshot: MarketSnapshot) -> MarketFeatures:
    """Build lesson-02 features from the two most recent closed candles."""
    if len(snapshot.candles) < 2:
        raise ValueError("At least two closed candles are required")

    previous = snapshot.candles[-2]
    current = snapshot.candles[-1]

    return_1 = (current.close - previous.close) / previous.close
    candle_range = current.high - current.low
    candle_body = current.close - current.open
    body_to_range = candle_body / candle_range if candle_range > 0 else 0.0

    return MarketFeatures(
        return_1=return_1,
        candle_range=candle_range,
        candle_body=candle_body,
        body_to_range=body_to_range,
    )
