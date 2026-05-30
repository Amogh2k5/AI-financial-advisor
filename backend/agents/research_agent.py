"""
research_agent.py — Market Research Agent for the AI Wealth Advisor.

Responsibility:
  Fetch live equity data (price, fundamentals, sector) for specific tickers
  that the user mentioned.  Feeds its findings into the coordinator's context.

Dedicated tool:
  - get_stock_price

Routing trigger (coordinator uses this agent when):
  - The user asks about specific stocks or wants equity research.
  - Keywords: price, analyse, ticker, RELIANCE, TCS, NVDA, stock, etc.
"""

import json
from typing import Any, Dict, List, Optional

from ..tools import TOOLS_SCHEMA
from .config import MAX_STEPS
from .prompts import research_system_prompt
from .utility_agent import (
    call_llm,
    execute_tool,
    extract_final_answer,
    parse_json_response,
)


# ─────────────────────────────────────────────
#  Tool schema slice  (only tools this agent uses)
# ─────────────────────────────────────────────

_RESEARCH_TOOLS: List[Dict[str, Any]] = [
    schema for schema in TOOLS_SCHEMA if schema["name"] in {"get_stock_price"}
]


# ─────────────────────────────────────────────
#  ResearchAgent
# ─────────────────────────────────────────────

class ResearchAgent:
    """
    Specialised agent for live equity research.

    It runs an agentic loop limited to `get_stock_price` calls, collecting
    data for every ticker of interest.  The aggregated result is returned
    as a structured dict consumed by the CoordinatorAgent.
    """

    def __init__(self) -> None:
        """Initialise the ResearchAgent with empty state."""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.steps: int = 0

    def reset(self) -> None:
        """Reset all state for a fresh analysis run."""
        self.tool_calls = []
        self.tool_results = {}
        self.steps = 0

    def run(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Fetch live data for each requested ticker.

        Unlike the coordinator, this agent bypasses the full LLM loop for
        efficiency — it calls `get_stock_price` directly for each ticker
        since the mapping is deterministic.

        Args:
            tickers: List of ticker symbols to research (e.g. ["RELIANCE.NS", "NVDA"]).

        Returns:
            A dict mapping each ticker to its price/fundamental data dict.
        """
        self.reset()
        results: Dict[str, Any] = {}

        print(f"\n\033[94m[ResearchAgent] Fetching data for: {tickers}\033[0m")

        for ticker in tickers:
            self.steps += 1
            full_result, _ = execute_tool("get_stock_price", {"ticker": ticker})
            results[ticker] = full_result
            self.tool_calls.append(
                {"tool": "get_stock_price", "params": {"ticker": ticker}, "step": self.steps}
            )
            self.tool_results[ticker] = full_result

            if "error" in full_result:
                print(f"  ⚠️  [ResearchAgent] Could not fetch {ticker}: {full_result['error']}")
            else:
                print(
                    f"  ✅  [ResearchAgent] {ticker}: "
                    f"₹{full_result.get('price_usd', 'N/A')} "
                    f"({full_result.get('change_pct', 'N/A')}%)"
                )

        return results

    def run_llm_loop(self, query: str) -> str:
        """
        Run a full LLM-driven research loop for open-ended research queries.

        Use this when the tickers are not known in advance and need to be
        inferred from the user's natural-language query.

        Args:
            query: Natural-language research request.

        Returns:
            Final answer string from the research agent's LLM loop.
        """
        self.reset()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": research_system_prompt(_RESEARCH_TOOLS, MAX_STEPS)},
            {"role": "user", "content": query},
        ]

        final_answer: Optional[str] = None

        while self.steps < MAX_STEPS:
            self.steps += 1
            raw = call_llm(messages)
            parsed = parse_json_response(raw)
            action = parsed.get("action", "error")

            if action == "final_answer":
                final_answer = extract_final_answer(parsed.get("answer", raw))
                break

            if action == "error":
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON. "
                        "Please respond with ONLY a valid JSON object."
                    ),
                })
                continue

            params = parsed.get("params", {})
            self.tool_calls.append({"tool": action, "params": params, "step": self.steps})

            full_result, llm_summary = execute_tool(action, params)
            self.tool_results[action] = full_result

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{action}' returned:\n{llm_summary}\n\n"
                    "Continue your analysis or provide the final_answer."
                ),
            })

        return final_answer or "Research complete. No additional equity analysis needed."
