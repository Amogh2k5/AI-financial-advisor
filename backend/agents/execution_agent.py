"""
execution_agent.py — Portfolio Construction Agent for the AI Wealth Advisor.

Responsibility:
  Translate a computed risk profile into a concrete, named portfolio of investment
  instruments with specific INR allocations and rationale for each pick.

Dedicated tool:
  - suggest_portfolio

Routing trigger (coordinator uses this agent when):
  - A concrete portfolio needs to be built.
  - A risk profile has already been computed by the AnalyticsAgent.
  - Keywords: portfolio, allocate, invest, recommend stocks, build plan, etc.
"""

import json
from typing import Any, Dict, List, Optional

from ..tools import TOOLS_SCHEMA
from .config import MAX_STEPS
from .prompts import execution_system_prompt
from .utility_agent import (
    call_llm,
    execute_tool,
    extract_final_answer,
    parse_json_response,
)


# ─────────────────────────────────────────────
#  Tool schema slice
# ─────────────────────────────────────────────

_EXECUTION_TOOLS: List[Dict[str, Any]] = [
    schema for schema in TOOLS_SCHEMA if schema["name"] in {"suggest_portfolio"}
]


# ─────────────────────────────────────────────
#  ExecutionAgent
# ─────────────────────────────────────────────

class ExecutionAgent:
    """
    Specialised agent for building a concrete investment portfolio.

    Provides a direct helper for deterministic portfolio construction (used by
    the coordinator in the fast path) and a full LLM loop for cases where the
    model needs to reason about custom allocations.
    """

    def __init__(self) -> None:
        """Initialise the ExecutionAgent with empty state."""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.steps: int = 0

    def reset(self) -> None:
        """Reset all state for a fresh run."""
        self.tool_calls = []
        self.tool_results = {}
        self.steps = 0

    # ── Direct helper (called deterministically by the coordinator) ───────────

    def build_portfolio(
        self,
        budget: float,
        risk_level: str,
        goals: str = "",
        custom_allocations: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a concrete portfolio directly (no LLM overhead).

        Args:
            budget:             Total investable amount in INR.
            risk_level:         One of "low", "medium", or "high".
            goals:              The investor's stated financial goals.
            custom_allocations: Optional JSON string of custom instrument picks,
                                e.g. '[{"symbol":"TCS.NS","weight_pct":25,"why":"..."}]'.

        Returns:
            Portfolio dict with 'instruments', 'total_budget_inr', and 'risk_level'.
        """
        self.steps += 1
        params: Dict[str, Any] = {
            "budget": budget,
            "risk_level": risk_level,
            "goals": goals,
        }
        if custom_allocations:
            params["custom_allocations"] = custom_allocations

        full_result, _ = execute_tool("suggest_portfolio", params)
        self.tool_calls.append({"tool": "suggest_portfolio", "params": params, "step": self.steps})
        self.tool_results["suggest_portfolio"] = full_result

        instruments = full_result.get("instruments", [])
        print(
            f"\n\033[92m[ExecutionAgent] ✅ Portfolio built — "
            f"{len(instruments)} instruments | "
            f"Budget: ₹{budget:,.0f} | Risk: {risk_level.capitalize()}\033[0m"
        )
        return full_result

    def get_portfolio_symbols(self) -> List[str]:
        """
        Extract the list of ticker symbols from the last built portfolio.

        Excludes non-tradeable placeholders (CASH, GOLD) that cannot be passed
        to yfinance-based trend tools.

        Returns:
            List of ticker symbols from the most recent portfolio, or [].
        """
        portfolio = self.tool_results.get("suggest_portfolio", {})
        return [
            inst.get("symbol")
            for inst in portfolio.get("instruments", [])
            if inst.get("symbol") and inst.get("symbol") not in {"CASH", "GOLD"}
        ]

    # ── LLM instrument picker ─────────────────────────────────────────────────

    def pick_instruments_with_llm(
        self,
        budget: float,
        risk_level: str,
        goals: str,
        risk_profile: Dict[str, Any],
        market_snapshot: Dict[str, Any],
    ) -> Optional[str]:
        """
        Ask the LLM to select specific investment instruments based on the
        investor's full context.

        This replaces the hardcoded default portfolios in suggest_portfolio().
        The LLM chooses tickers, weights, and rationale; the result is passed
        as `custom_allocations` to build_portfolio().

        Args:
            budget:          Total investable amount in INR.
            risk_level:      One of "low", "medium", or "high".
            goals:           The investor's stated financial goals.
            risk_profile:    Output of calculate_risk() — allocation %, horizon, etc.
            market_snapshot: Output of get_market_summary() — live index prices.

        Returns:
            A JSON string of the form:
              '[{"symbol": "TCS.NS", "weight_pct": 25, "why": "..."},  ...]'
            or None if the LLM fails to produce valid output.
        """
        alloc = risk_profile.get("allocation", {})

        # Build a compact market context string
        market_lines = []
        for sym, data in market_snapshot.items():
            if isinstance(data, dict) and not data.get("error"):
                market_lines.append(
                    f"  {data.get('name', sym)}: "
                    f"{data.get('price', 'N/A')} "
                    f"({data.get('percent_change', 0):+.2f}%)"
                )
        market_ctx = "\n".join(market_lines) or "  (market data unavailable)"

        picker_prompt = f"""You are a senior portfolio manager specialising in Indian retail investors.

Select specific investment instruments for this investor:

## Investor Profile
- Budget: ₹{budget:,.0f}
- Risk Level: {risk_level.capitalize()}
- Goals: {goals}
- Age: {risk_profile.get('age', 'N/A')} years
- Investment Horizon: {risk_profile.get('investment_horizon', 'N/A')}
- Expected Annual Return: {risk_profile.get('expected_annual_return', 'N/A')}
- Max Drawdown Tolerance: {risk_profile.get('max_drawdown_pct', 'N/A')}%

## Recommended Asset Allocation
- Equity: {alloc.get('equity', 0)}%
- Bonds/Debt: {alloc.get('bonds', 0)}%
- Gold: {alloc.get('gold', 0)}%
- Cash: {alloc.get('cash', 0)}%

## Live Market Conditions
{market_ctx}

## Your Task
Return ONLY a valid JSON array of 4–6 instruments that best fit this profile.
Use real, yfinance-compatible symbols (e.g. RELIANCE.NS, TCS.NS, ^NSEI, NVDA, GC=F).
Weights must sum to 100. Include at least one defensive instrument.

Format — return ONLY this JSON, no other text:
[
  {{"symbol": "TICKER", "weight_pct": 30, "why": "one-line reason"}},
  ...
]"""

        print(f"\n\033[94m[ExecutionAgent] 🤖 Asking LLM to pick instruments for {risk_level} risk profile...\033[0m")

        try:
            messages = [
                {"role": "system", "content": "You are a portfolio construction expert. Return only valid JSON arrays."},
                {"role": "user", "content": picker_prompt},
            ]
            raw = call_llm(messages)

            # Strip markdown fences if present
            import re
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

            # Extract the JSON array
            array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if array_match:
                candidate = array_match.group()
                picks = json.loads(candidate)

                # Validate structure
                if isinstance(picks, list) and len(picks) >= 2:
                    valid = [
                        p for p in picks
                        if isinstance(p, dict)
                        and p.get("symbol")
                        and isinstance(p.get("weight_pct"), (int, float))
                    ]
                    if valid:
                        symbols = [p["symbol"] for p in valid]
                        print(f"\033[92m[ExecutionAgent] ✅ LLM picked: {symbols}\033[0m")
                        return json.dumps(valid)

        except Exception as exc:
            print(f"\033[93m[ExecutionAgent] ⚠️  LLM instrument picker failed: {exc} — using defaults\033[0m")

        return None  # Caller falls back to hardcoded defaults

    # ── LLM-driven loop (for custom/complex allocation requests) ──────────────

    def run_llm_loop(self, query: str) -> str:
        """
        Run a full LLM-driven loop for custom portfolio construction queries.

        Use this when the investor has specific preferences that require the
        model to reason about custom instrument picks.

        Args:
            query: Natural-language portfolio construction request.

        Returns:
            Final answer string from the execution LLM loop.
        """
        self.reset()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": execution_system_prompt(_EXECUTION_TOOLS, MAX_STEPS)},
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
                    "Continue building the portfolio or provide the final_answer."
                ),
            })

        return final_answer or "Portfolio construction completed."
