# -*- coding: utf-8 -*-
"""
api.py — FastAPI backend for the AI Wealth Advisor.

Endpoints:
  GET  /api/health          — Verify Ollama is reachable
  GET  /api/market          — Live market snapshot (no agent, instant)
  POST /api/analyze         — Full synchronous analysis (returns when done)
  POST /api/analyze/stream  — Real-time SSE stream of every agent step

Run with:
  uvicorn api:app --reload --port 8000
"""

import json
import queue
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import CoordinatorAgent  # Multi-agent orchestrator
from .tools import get_market_summary, TOOLS_SCHEMA

# ─────────────────────────────────────────────
#  App Initialisation
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Wealth Advisor API",
    description="Agentic financial planning system powered by Qwen 2.5 via Ollama.",
    version="1.0.0",
)

# Allow all origins so any frontend (React, HTML, etc.) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

OLLAMA_URL = "http://localhost:11434"


# ─────────────────────────────────────────────
#  Request / Response Models (Pydantic)
# ─────────────────────────────────────────────

class UserProfile(BaseModel):
    """The investor profile collected from the frontend form."""
    name:       str   = Field(...,  example="Amogh")
    age:        int   = Field(...,  ge=18, le=100, example=22)
    budget:     float = Field(...,  gt=0,  example=100000)
    risk_level: str   = Field(...,  example="medium",
                              description="One of: low, medium, high")
    goals:      str   = Field(...,  example="long-term wealth creation")
    stocks:     list[str] = Field(default=[], example=["RELIANCE.NS", "TCS.NS"])


class AnalysisResult(BaseModel):
    """Returned by /api/analyze (synchronous endpoint)."""
    answer:     str
    tool_calls: list[dict]
    steps:      int
    profile:    dict


# ─────────────────────────────────────────────
#  Helper: build agent query string
# ─────────────────────────────────────────────

def _build_query(profile: UserProfile) -> str:
    """Convert a UserProfile into one natural-language query for the agent."""
    budget_str = f"INR {profile.budget:,.0f}"
    stocks_str = (
        f"Also analyse these specific stocks for me: {', '.join(profile.stocks)}."
        if profile.stocks else ""
    )
    return (
        f"I am {profile.name}, {profile.age} years old. "
        f"I have {budget_str} to invest. "
        f"My risk tolerance is {profile.risk_level}. "
        f"My financial goals are: {profile.goals}. "
        f"{stocks_str} "
        f"Please: "
        f"1) Check the current market conditions, "
        f"2) Calculate my risk profile and suitable asset allocation, "
        f"3) Suggest a detailed portfolio with specific instruments and amounts. "
        f"Give me a complete, actionable financial plan."
    )


# ─────────────────────────────────────────────
#  GET /api/health
# ─────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health_check():
    """
    Verify that the FastAPI server is running.
    Returns status, provider, and current model.
    """
    from .agents.config import AGENT_MODEL, OPENROUTER_API_KEY
    
    provider = "OpenRouter" if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_key_here" else "Ollama"
    
    try:
        # Check Ollama connectivity regardless, for local tools
        ollama_status = "connected"
        try:
            http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        except:
            ollama_status = "unreachable"

        return {
            "status":         "ok",
            "provider":       provider,
            "agent_model":    AGENT_MODEL,
            "ollama_local":   ollama_status,
            "tools_available": [t["name"] for t in TOOLS_SCHEMA],
        }
    except Exception as e:
        return {
            "status":  "degraded",
            "detail":  str(e),
        }


# ─────────────────────────────────────────────
#  GET /api/market
# ─────────────────────────────────────────────

@app.get("/api/market", tags=["Market"])
def market_snapshot():
    """
    Fetch a live snapshot of key benchmarks:
    Nifty 50, Sensex, S&P 500, Gold (USD), USD/INR.
    This does NOT use the agent — it is instant.
    """
    try:
        data = get_market_summary()
        return {"status": "ok", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data fetch failed: {e}")


# ─────────────────────────────────────────────
#  POST /api/analyze   (synchronous)
# ─────────────────────────────────────────────

@app.post("/api/analyze", response_model=AnalysisResult, tags=["Analysis"])
def analyze(profile: UserProfile):
    """
    Run the full agentic analysis synchronously.
    Blocks until the agent completes (typically 30–90 seconds).
    Good for simple frontends that don't need live step updates.
    """
    query = _build_query(profile)

    try:
        agent  = CoordinatorAgent()
        answer = agent.run(query)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return AnalysisResult(
        answer=answer,
        tool_calls=agent.tool_calls,
        steps=agent.steps,
        profile=profile.model_dump(),
    )


# ─────────────────────────────────────────────
#  POST /api/analyze/stream   (SSE)
# ─────────────────────────────────────────────

@app.post("/api/analyze/stream", tags=["Analysis"])
def analyze_stream(profile: UserProfile):
    """
    Run the agentic analysis and stream every step as a
    Server-Sent Event (SSE). The frontend can display the
    agent's reasoning in real-time using the EventSource API.

    SSE Event format:
      data: {"type": "thinking"|"tool_call"|"tool_result"|"final_answer"|"error"|"done", ...}

    Event types:
      thinking     — agent is calling Ollama
      tool_call    — agent decided to call a tool
      tool_result  — tool returned a result
      final_answer — agent's final recommendation (last event before "done")
      error        — something went wrong
      done         — stream is finished (always sent last)
    """
    # Thread-safe queue: agent thread pushes events, SSE generator pops them
    event_q: queue.Queue = queue.Queue()

    query = _build_query(profile)

    def run_agent_thread():
        """Run the agent in a background thread so we don't block the HTTP handler."""
        try:
            agent = CoordinatorAgent(event_queue=event_q)
            answer = agent.run(query)
            # Push a "done" sentinel with the summary
            event_q.put({
                "type":       "done",
                "answer":     answer,
                "tool_calls": agent.tool_calls,
                "tool_results": getattr(agent, 'tool_results', {}),
                "steps":      agent.steps,
            })
        except ConnectionError as e:
            event_q.put({"type": "error", "content": str(e)})
            event_q.put({"type": "done", "answer": "", "tool_calls": [], "steps": 0})
        except Exception as e:
            event_q.put({"type": "error", "content": f"Agent crashed: {e}"})
            event_q.put({"type": "done", "answer": "", "tool_calls": [], "steps": 0})

    # Kick off the agent thread
    thread = threading.Thread(target=run_agent_thread, daemon=True)
    thread.start()

    def sse_generator():
        """
        Generator that reads events from the queue and yields them
        as SSE-formatted strings. Stops when it sees the "done" event.
        """
        while True:
            try:
                # Wait up to 120 seconds for the next event
                event = event_q.get(timeout=120)
            except queue.Empty:
                # Timeout — send a heartbeat and keep waiting
                yield "data: {\"type\": \"heartbeat\"}\n\n"
                continue

            # Serialise to JSON and yield as SSE
            # Use allow_nan=False to catch NaN early, fall back to a sanitised version
            try:
                payload = json.dumps(event, ensure_ascii=False, default=str, allow_nan=False)
            except ValueError:
                # Sanitise: replace NaN/Infinity with None (null in JSON)
                import math
                def sanitise(obj):
                    if isinstance(obj, float):
                        return None if (math.isnan(obj) or math.isinf(obj)) else obj
                    if isinstance(obj, dict):
                        return {k: sanitise(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [sanitise(v) for v in obj]
                    return obj
                payload = json.dumps(sanitise(event), ensure_ascii=False, default=str)
            yield f"data: {payload}\n\n"

            # "done" is always the last event
            if event.get("type") == "done":
                break

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if behind a proxy
        },
    )


# ─────────────────────────────────────────────
#  GET /   (API root)
# ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def redirect_to_app():
    return RedirectResponse(url="/app")

@app.get("/api/info", tags=["System"])
def root():
    return {
        "message": "AI Wealth Advisor API is running.",
        "docs":    "http://localhost:8000/docs",
        "frontend": "http://localhost:8000/app/"
    }
