from dataclasses import dataclass

from metatrader_llm_lab.bridge.schemas import MarketSnapshot


@dataclass(frozen=True)
class MarketFeatures:
    """Small, explainable numerical features derived from closed candles."""

    return_1: float
    return_3: float
    return_5: float
    return_10: float
    candle_range: float
    candle_body: float
    body_to_range: float


def _return_over_bars(snapshot: MarketSnapshot, bars: int) -> float:
    """Return from a past close to the latest close."""
    current_close = snapshot.candles[-1].close
    past_close = snapshot.candles[-1 - bars].close
    return (current_close - past_close) / past_close


def build_market_features(snapshot: MarketSnapshot) -> MarketFeatures:
    """Build lesson-02 features from recent closed-candle history."""
    if len(snapshot.candles) < 11:
        raise ValueError("At least 11 closed candles are required")

    current = snapshot.candles[-1]

    candle_range = current.high - current.low
    candle_body = current.close - current.open
    body_to_range = candle_body / candle_range if candle_range > 0 else 0.0

    return MarketFeatures(
        return_1=_return_over_bars(snapshot, 1),
        return_3=_return_over_bars(snapshot, 3),
        return_5=_return_over_bars(snapshot, 5),
        return_10=_return_over_bars(snapshot, 10),
        candle_range=candle_range,
        candle_body=candle_body,
        body_to_range=body_to_range,
    )
