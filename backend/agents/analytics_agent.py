"""
analytics_agent.py — Risk & Analytics Agent for the AI Wealth Advisor.

Responsibility:
  Compute a quantitative risk profile from the investor's budget, age, and risk
  tolerance.  Returns allocation percentages and portfolio constraints that feed
  directly into the ExecutionAgent.

Dedicated tool:
  - calculate_risk

Routing trigger (coordinator uses this agent when):
  - A risk profile needs to be computed.
  - Keywords: risk, allocation, conservative, aggressive, age, budget, etc.
"""

import json
from typing import Any, Dict, List, Optional

from ..tools import TOOLS_SCHEMA
from .config import MAX_STEPS
from .prompts import analytics_system_prompt
from .utility_agent import (
    call_llm,
    execute_tool,
    extract_final_answer,
    parse_json_response,
)


# ─────────────────────────────────────────────
#  Tool schema slice
# ─────────────────────────────────────────────

_ANALYTICS_TOOLS: List[Dict[str, Any]] = [
    schema for schema in TOOLS_SCHEMA if schema["name"] in {"calculate_risk"}
]


# ─────────────────────────────────────────────
#  AnalyticsAgent
# ─────────────────────────────────────────────

class AnalyticsAgent:
    """
    Specialised agent for investor risk profiling and asset allocation analytics.

    Provides a direct helper for deterministic risk calculation and a full LLM
    loop for complex, multi-step analytical queries.
    """

    def __init__(self) -> None:
        """Initialise the AnalyticsAgent with empty state."""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.steps: int = 0

    def reset(self) -> None:
        """Reset all state for a fresh analysis run."""
        self.tool_calls = []
        self.tool_results = {}
        self.steps = 0

    # ── Direct helper (called deterministically by the coordinator) ───────────

    def compute_risk_profile(
        self,
        budget: float,
        risk_level: str,
        age: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute the investor's risk profile directly (no LLM overhead).

        Args:
            budget:     Total investable amount in INR.
            risk_level: One of "low", "medium", or "high".
            age:        Investor's age (default 30).

        Returns:
            Risk profile dict including allocation percentages, risk score,
            expected return, and investment horizon.
        """
        self.steps += 1
        params: Dict[str, Any] = {
            "budget": budget,
            "risk_level": risk_level,
            "age": age,
        }
        full_result, _ = execute_tool("calculate_risk", params)
        self.tool_calls.append({"tool": "calculate_risk", "params": params, "step": self.steps})
        self.tool_results["calculate_risk"] = full_result

        alloc = full_result.get("allocation", {})
        print(
            f"\n\033[92m[AnalyticsAgent] ✅ Risk profile computed — "
            f"{full_result.get('risk_level', 'N/A').capitalize()} | "
            f"Equity: {alloc.get('equity', 'N/A')}% | "
            f"Return: {full_result.get('expected_annual_return', 'N/A')}\033[0m"
        )
        return full_result

    # ── LLM-driven loop ────────────────────────────────────────────────────────

    def run_llm_loop(self, query: str) -> str:
        """
        Run a full LLM-driven analytics loop for complex, open-ended queries.

        Args:
            query: Natural-language analytics request.

        Returns:
            Final answer string from the analytics LLM loop.
        """
        self.reset()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": analytics_system_prompt(_ANALYTICS_TOOLS, MAX_STEPS)},
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

        return final_answer or "Risk analysis completed."
