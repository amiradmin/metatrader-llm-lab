from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MarketData(BaseModel):
    bid: float
    ask: float
    spread_points: float = Field(ge=0)


class CandleData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = Field(ge=0)


class AccountData(BaseModel):
    balance: float
    equity: float
    free_margin: float


class PositionData(BaseModel):
    status: Literal["OPEN", "NONE"]
    type: Literal["BUY", "SELL", "NONE"] = "NONE"
    volume: float = Field(ge=0)
    open_price: float
    profit: float


class MarketSnapshot(BaseModel):
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    market: MarketData
    last_closed_candle: CandleData
    account: AccountData
    position: PositionData
    timestamp: datetime


class TradeDecision(BaseModel):
    decision: Literal["BUY", "SELL", "WAIT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
