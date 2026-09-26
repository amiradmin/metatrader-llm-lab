"""Lesson 03: turn numerical market state into compact LLM context."""

from metatrader_llm_lab.bridge.schemas import MarketSnapshot
from metatrader_llm_lab.features.market import MarketFeatures


def build_market_context(
    snapshot: MarketSnapshot,
    features: MarketFeatures,
) -> str:
    """Create deterministic text context for a language model."""
    position = snapshot.position

    return "\n".join(
        [
            f"symbol: {snapshot.symbol}",
            f"timeframe: {snapshot.timeframe}",
            f"spread_points: {snapshot.market.spread_points:.1f}",
            f"return_1_pct: {features.return_1 * 100:+.4f}",
            f"return_3_pct: {features.return_3 * 100:+.4f}",
            f"return_5_pct: {features.return_5 * 100:+.4f}",
            f"return_10_pct: {features.return_10 * 100:+.4f}",
            f"candle_body_to_range: {features.body_to_range:+.4f}",
            f"average_range_5: {features.average_range_5:.4f}",
            f"volume_ratio_5: {features.volume_ratio_5:.4f}",
            f"position_status: {position.status}",
            f"position_type: {position.type}",
        ]
    )
