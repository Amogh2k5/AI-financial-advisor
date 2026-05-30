"""
config.py — Centralised configuration for the AI Wealth Advisor multi-agent system.

All environment variables, model settings, and shared constants are defined here.
Individual agent modules import from this file rather than reading env vars directly.
"""

import os
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv(override=True)


# ─────────────────────────────────────────────
#  Provider / Model Settings  (DO NOT CHANGE)
# ─────────────────────────────────────────────

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

# Fallback local model (used only when no OpenRouter key is present)
OLLAMA_URL: str = "http://localhost:11434/api/chat"

# The model to use — read from .env, e.g. "google/gemma-2-27b-it:free"
AGENT_MODEL: str = os.getenv("MODEL_NAME", "qwen2.5:7b")


# ─────────────────────────────────────────────
#  Agent Loop Settings
# ─────────────────────────────────────────────

MAX_STEPS: int = 10          # Maximum tool-call iterations per agent
LLM_TIMEOUT: int = 60        # Seconds before an LLM request times out
LLM_TEMPERATURE: float = 0.3 # Sampling temperature for all agents


# ─────────────────────────────────────────────
#  Routing Keys  (used by the coordinator)
# ─────────────────────────────────────────────

AGENT_RESEARCH: str   = "research"
AGENT_RETRIEVAL: str  = "retrieval"
AGENT_ANALYTICS: str  = "analytics"
AGENT_EXECUTION: str  = "execution"
AGENT_UTILITY: str    = "utility"


# ─────────────────────────────────────────────
#  Convenience helpers
# ─────────────────────────────────────────────

def is_openrouter_configured() -> bool:
    """Return True when a valid OpenRouter API key is present."""
    return bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_key_here")


def active_provider() -> str:
    """Return a human-readable name of the active LLM provider."""
    return "OpenRouter" if is_openrouter_configured() else "Ollama"
