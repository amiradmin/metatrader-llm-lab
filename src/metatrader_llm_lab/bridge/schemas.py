from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    """A point-in-time view of the market sent by MetaTrader 5."""

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    bid: float
    ask: float
    spread_points: float = Field(ge=0)
    position: str = "NONE"
    timestamp: datetime


class TradeDecision(BaseModel):
    """Structured response returned to MetaTrader 5."""

    decision: Literal["BUY", "SELL", "WAIT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
