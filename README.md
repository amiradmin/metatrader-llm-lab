# MetaTrader LLM Lab

A hands-on learning project for building an AI-assisted MetaTrader 5 trading robot while revisiting core LLM concepts step by step.

## Lesson 01 — Architecture and Market Snapshot

Goal: build a reliable data path from MetaTrader 5 to Python before connecting a real AI model.

```text
MetaTrader 5
    -> Market Snapshot
    -> Python Bridge
    -> AI Engine
    -> Decision JSON
```

The first AI engine intentionally returns `WAIT` so the transport path can be tested safely.

## Initial structure

```text
metatrader-llm-lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── mt5/
│   └── HectorLLMLab.mq5
├── src/
│   └── metatrader_llm_lab/
│       ├── __init__.py
│       ├── bridge/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── schemas.py
│       └── ai/
│           ├── __init__.py
│           └── engine.py
└── tests/
    └── test_bridge.py
```

## Run the bridge

```bash
uv sync
uv run uvicorn metatrader_llm_lab.bridge.app:app --host 127.0.0.1 --port 8010
```

Health check:

```bash
curl -s http://127.0.0.1:8010/health | jq
```

Snapshot test:

```bash
curl -s -X POST http://127.0.0.1:8010/snapshot \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol":"XAUUSD_l",
    "timeframe":"M15",
    "bid":4353.67,
    "ask":4354.09,
    "spread_points":42,
    "position":"NONE",
    "timestamp":"2026-09-22T07:15:00"
  }' | jq
```

Expected decision:

```json
{
  "decision": "WAIT",
  "confidence": 0.0,
  "reason": "AI model is not connected yet"
}
```
