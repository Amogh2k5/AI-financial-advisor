"""
retrieval_agent.py — Market Data Retrieval Agent for the AI Wealth Advisor.

Responsibility:
  Fetch live market snapshots and 30-day historical trend data for benchmarks
  and portfolio symbols.  Provides the raw data used to render frontend charts.

Dedicated tools:
  - get_market_summary   : Real-time benchmark snapshot
  - get_market_trends    : 30-day historical performance trends

Routing trigger (coordinator uses this agent when):
  - A live market overview is needed.
  - Historical trend data is needed for chart generation.
"""

import json
from typing import Any, Dict, List, Optional

from ..tools import TOOLS_SCHEMA
from .config import MAX_STEPS
from .prompts import retrieval_system_prompt
from .utility_agent import (
    call_llm,
    execute_tool,
    extract_final_answer,
    parse_json_response,
)


# ─────────────────────────────────────────────
#  Tool schema slice
# ─────────────────────────────────────────────

_RETRIEVAL_TOOLS: List[Dict[str, Any]] = [
    schema
    for schema in TOOLS_SCHEMA
    if schema["name"] in {"get_market_summary", "get_market_trends"}
]


# ─────────────────────────────────────────────
#  RetrievalAgent
# ─────────────────────────────────────────────

class RetrievalAgent:
    """
    Specialised agent for fetching market data and historical trends.

    Provides two direct-call helpers for the coordinator (no LLM overhead)
    and a full LLM loop for open-ended retrieval queries.
    """

    def __init__(self) -> None:
        """Initialise the RetrievalAgent with empty state."""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.steps: int = 0

    def reset(self) -> None:
        """Reset all state for a fresh run."""
        self.tool_calls = []
        self.tool_results = {}
        self.steps = 0

    # ── Direct helpers (called deterministically by the coordinator) ──────────

    def fetch_market_snapshot(self) -> Dict[str, Any]:
        """
        Fetch a live snapshot of key benchmark indices.

        Returns:
            Raw dict from get_market_summary (Nifty 50, Sensex, S&P 500,
            Gold USD, USD/INR).
        """
        self.steps += 1
        full_result, _ = execute_tool("get_market_summary", {})
        self.tool_calls.append(
            {"tool": "get_market_summary", "params": {}, "step": self.steps}
        )
        self.tool_results["get_market_summary"] = full_result
        print("\n\033[92m[RetrievalAgent] ✅ Market snapshot fetched.\033[0m")
        return full_result

    def fetch_trends(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch 30-day historical percentage return trends for a list of symbols.

        Benchmarks (^NSEI, ^GSPC) are automatically prepended if not present.

        Args:
            symbols: List of ticker symbols, e.g. ["RELIANCE.NS", "NVDA"].

        Returns:
            Raw dict from get_market_trends with 'dates', 'trends', and 'names'.
        """
        # Always include the main benchmarks for comparison
        base = ["^NSEI", "^GSPC"]
        all_symbols = list(dict.fromkeys(base + symbols))  # unique, order-preserved

        symbols_str = ",".join(all_symbols)
        self.steps += 1
        full_result, _ = execute_tool("get_market_trends", {"symbols": symbols_str})
        self.tool_calls.append(
            {"tool": "get_market_trends", "params": {"symbols": symbols_str}, "step": self.steps}
        )
        self.tool_results["get_market_trends"] = full_result
        print(
            f"\n\033[92m[RetrievalAgent] ✅ Trend data fetched for: {all_symbols}\033[0m"
        )
        return full_result

    # ── LLM-driven loop (for open-ended retrieval queries) ─────────────────────

    def run_llm_loop(self, query: str) -> str:
        """
        Run a full LLM-driven loop for open-ended market-data queries.

        Args:
            query: Natural-language market data request.

        Returns:
            Final answer string.
        """
        self.reset()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": retrieval_system_prompt(_RETRIEVAL_TOOLS, MAX_STEPS)},
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
                    "Continue your retrieval or provide the final_answer."
                ),
            })

        return final_answer or "Market data retrieved successfully."
