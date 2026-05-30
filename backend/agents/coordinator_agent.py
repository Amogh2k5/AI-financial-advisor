"""
coordinator_agent.py — Orchestrator for the AI Wealth Advisor multi-agent system.

The CoordinatorAgent is the single entry point that routes work to:
  1. RetrievalAgent  — market snapshot (get_market_summary)
  2. ResearchAgent   — specific stock lookup (get_stock_price)
  3. AnalyticsAgent  — risk profiling (calculate_risk)
  4. ExecutionAgent  — portfolio construction (suggest_portfolio)
  5. RetrievalAgent  — trend data for charts (get_market_trends)
  6. LLM final pass  — write the comprehensive Markdown report

If the LLM's final answer is unusable, UtilityAgent generates a structured
Markdown fallback deterministically from the collected tool results.

Routing Flow:
  user_query
    └─→ [RetrievalAgent]  fetch_market_snapshot()        ← always first
    └─→ [ResearchAgent]   run(tickers)                   ← if user mentioned stocks
    └─→ [AnalyticsAgent]  compute_risk_profile()         ← always
    └─→ [ExecutionAgent]  build_portfolio()              ← always
    └─→ [RetrievalAgent]  fetch_trends(portfolio_syms)  ← always (for charts)
    └─→ LLM              synthesise final report
    └─→ [UtilityAgent]   build_fallback_report()         ← if LLM answer is garbage

Public API mirrors the original WealthAdvisorAgent:
  agent = CoordinatorAgent(event_queue=None)
  answer = agent.run(query)
  agent.tool_results   # dict used by api.py for chart data
  agent.tool_calls     # list of all tool invocations
  agent.steps          # total step count
"""

import json
import re
from queue import Queue
from typing import Any, Dict, List, Optional

from ..tools import TOOLS_SCHEMA
from .analytics_agent import AnalyticsAgent
from .config import (
    AGENT_MODEL,
    MAX_STEPS,
    active_provider,
    is_openrouter_configured,
)
from .execution_agent import ExecutionAgent
from .prompts import coordinator_system_prompt
from .research_agent import ResearchAgent
from .retrieval_agent import RetrievalAgent
from .utility_agent import (
    UtilityAgent,
    call_llm,
    execute_tool,
    extract_final_answer,
    parse_json_response,
    summarise_tool_result_for_llm,
)


# ─────────────────────────────────────────────
#  CoordinatorAgent
# ─────────────────────────────────────────────

class CoordinatorAgent:
    """
    Orchestrator that delegates to specialised sub-agents and synthesises
    their results into a comprehensive financial recommendation.

    This class is a drop-in replacement for the original WealthAdvisorAgent —
    it exposes the same public interface (run, tool_calls, tool_results, steps)
    so that api.py requires zero changes.

    Args:
        event_queue: An optional thread-safe queue.Queue.
                     When provided, every agent step pushes a structured event
                     dict for SSE streaming.  When None, events are only printed
                     to stdout (CLI mode).
    """

    # Label → frontend event_type mapping (same as original agent)
    _LABEL_TO_TYPE: Dict[str, str] = {
        "🤔 Thinking":    "thinking",
        "🔧 Tool Call":   "tool_call",
        "📊 Tool Result": "tool_result",
        "✅ Final Answer": "final_answer",
        "❌ Error":       "error",
    }

    def __init__(self, event_queue: Optional[Queue] = None) -> None:
        """Initialise the CoordinatorAgent and all sub-agents."""
        self.event_queue: Optional[Queue] = event_queue

        # Sub-agents
        self._retrieval = RetrievalAgent()
        self._research  = ResearchAgent()
        self._analytics = AnalyticsAgent()
        self._execution = ExecutionAgent()
        self._utility   = UtilityAgent()

        # Shared state (exposed to api.py)
        self.tool_calls:   List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any]       = {}
        self.steps:        int                  = 0

    # ── State management ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset coordinator and all sub-agent state for a fresh run."""
        self.tool_calls   = []
        self.tool_results = {}
        self.steps        = 0
        self._retrieval.reset()
        self._research.reset()
        self._analytics.reset()
        self._execution.reset()
        # NOTE: event_queue is intentionally NOT reset

    # ── Step logger / event emitter ───────────────────────────────────────────

    def _print_step(self, step: int, label: str, content: str) -> None:
        """
        Print a step to stdout (CLI) and optionally push an SSE event.

        Args:
            step:    The current step number.
            label:   A human-readable label (e.g. "🤔 Thinking").
            content: The step content to display / stream.
        """
        colours: Dict[str, str] = {
            "🤔 Thinking":    "\033[94m",
            "🔧 Tool Call":   "\033[93m",
            "📊 Tool Result": "\033[92m",
            "✅ Final Answer": "\033[96m",
            "❌ Error":       "\033[91m",
        }
        colour = colours.get(label, "")
        reset  = "\033[0m"

        print(f"\n{colour}{'─'*60}")
        print(f"  Step {step} | {label}")
        print(f"{'─'*60}{reset}")
        display = content if len(content) < 800 else content[:800] + "\n  ... [truncated]"
        print(display)

        if self.event_queue is not None:
            self.event_queue.put({
                "type":    self._LABEL_TO_TYPE.get(label, "info"),
                "step":    step,
                "label":   label,
                "content": content,
            })

    # ── Sub-agent result merger ───────────────────────────────────────────────

    def _merge_sub_agent_results(self) -> None:
        """
        Merge tool_calls and tool_results from all sub-agents into the
        coordinator's own collections.  This keeps api.py's single-dict
        interface intact.
        """
        for agent in (self._retrieval, self._research, self._analytics, self._execution):
            self.tool_calls.extend(agent.tool_calls)
            self.tool_results.update(agent.tool_results)

    # ── Ticker extractor ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_tickers(query: str) -> List[str]:
        """
        Extract explicitly mentioned ticker symbols from a user query.

        Looks for patterns like RELIANCE.NS, TCS.NS, NVDA, AAPL (all-caps words
        optionally suffixed with .NS / .BO).

        Args:
            query: The raw user query string.

        Returns:
            List of unique ticker strings found in the query.
        """
        pattern = r"\b([A-Z]{2,10}(?:\.[A-Z]{1,3})?)\b"
        # Exclude common English abbreviations and our own keywords
        stopwords = {
            "I", "INR", "USD", "MY", "SIP", "ETF", "FD", "NRI",
            "AI", "LLM", "API", "IPO", "RBI", "NSE", "BSE",
        }
        found = re.findall(pattern, query)
        return [t for t in dict.fromkeys(found) if t not in stopwords]

    # ── Orchestration pipeline ────────────────────────────────────────────────

    def _parse_profile(self, query: str) -> Dict[str, Any]:
        """
        Extract structured investor profile fields from the user query.

        Uses simple regex patterns rather than an LLM call to keep this
        fast and deterministic.

        Args:
            query: Natural-language user query.

        Returns:
            Dict with 'budget', 'risk_level', 'age', 'goals' (best-effort).
        """
        profile: Dict[str, Any] = {
            "budget":     100_000.0,
            "risk_level": "medium",
            "age":        30,
            "goals":      "long-term wealth creation",
        }

        # Budget — looks for INR / ₹ followed by a number
        budget_m = re.search(
            r"(?:INR|₹|Rs\.?)\s*([\d,]+(?:\.\d+)?)", query, re.IGNORECASE
        )
        if budget_m:
            profile["budget"] = float(budget_m.group(1).replace(",", ""))

        # Age
        age_m = re.search(r"\b(\d{2})\s*(?:years?\s*old|yr)", query, re.IGNORECASE)
        if age_m:
            profile["age"] = int(age_m.group(1))

        # Risk level
        if re.search(r"\blow\b", query, re.IGNORECASE):
            profile["risk_level"] = "low"
        elif re.search(r"\bhigh\b", query, re.IGNORECASE):
            profile["risk_level"] = "high"
        elif re.search(r"\bmedium\b|\bmoderate\b|\bbalanced\b", query, re.IGNORECASE):
            profile["risk_level"] = "medium"

        # Goals — capture everything after "goals are:" or "goal is:"
        goals_m = re.search(r"goals?\s+(?:are|is)[:\s]+(.+?)(?:\.|$)", query, re.IGNORECASE)
        if goals_m:
            profile["goals"] = goals_m.group(1).strip()

        return profile

    def _build_llm_context(self) -> str:
        """
        Serialise collected tool results into a compact JSON block to inject
        into the LLM's synthesis prompt.

        Returns:
            Formatted string summarising all gathered data.
        """
        context_parts: List[str] = []

        if "get_market_summary" in self.tool_results:
            compact = {
                sym: {
                    "name": d.get("name"),
                    "price": d.get("price"),
                    "change_pct": d.get("percent_change"),
                }
                for sym, d in self.tool_results["get_market_summary"].items()
                if isinstance(d, dict)
            }
            context_parts.append(
                f"## Live Market Snapshot\n{json.dumps(compact, indent=2, default=str)}"
            )

        if "calculate_risk" in self.tool_results:
            context_parts.append(
                f"## Risk Profile\n"
                f"{json.dumps(self.tool_results['calculate_risk'], indent=2, default=str)}"
            )

        if "suggest_portfolio" in self.tool_results:
            context_parts.append(
                f"## Recommended Portfolio\n"
                f"{json.dumps(self.tool_results['suggest_portfolio'], indent=2, default=str)}"
            )

        if self._research.tool_results:
            context_parts.append(
                f"## Stock Research\n"
                f"{json.dumps(self._research.tool_results, indent=2, default=str)}"
            )

        if "get_market_trends" in self.tool_results:
            trends = self.tool_results["get_market_trends"]
            summary = {
                sym: {
                    "start_pct": vals[0],
                    "end_pct": vals[-1],
                    "total_return_pct": round(vals[-1] - vals[0], 2),
                }
                for sym, vals in trends.get("trends", {}).items()
                if isinstance(vals, list) and vals
            }
            context_parts.append(
                f"## 30-Day Performance Trends\n{json.dumps(summary, indent=2)}"
            )

        return "\n\n".join(context_parts)

    # ── Main orchestration entry point ────────────────────────────────────────

    def run(self, user_query: str) -> str:
        """
        Execute the full multi-agent financial planning pipeline.

        Pipeline steps:
          1. Parse investor profile from the query.
          2. RetrievalAgent: fetch live market snapshot.
          3. ResearchAgent:  fetch data for user-mentioned stocks (if any).
          4. AnalyticsAgent: compute risk profile.
          5. ExecutionAgent: build portfolio.
          6. RetrievalAgent: fetch 30-day trends for portfolio symbols + benchmarks.
          7. LLM synthesis:  generate the final Markdown report.
          8. UtilityAgent:   fallback if LLM output is unusable.

        Args:
            user_query: The natural-language investor request.

        Returns:
            A comprehensive, Markdown-formatted financial recommendation string.
        """
        self.reset()

        print(f"\n\033[95m{'═'*60}")
        print(f"  🤖  AI Wealth Advisor — Multi-Agent System Started")
        print(f"{'═'*60}\033[0m")
        print(f"  Query: {user_query}\n")

        # ── Step 1: Parse investor profile ────────────────────────────────────
        self.steps += 1
        profile = self._parse_profile(user_query)
        self._print_step(
            self.steps, "🤔 Thinking",
            f"  Parsed profile: Budget=₹{profile['budget']:,.0f} | "
            f"Risk={profile['risk_level']} | Age={profile['age']}"
        )

        # ── Step 2: Market snapshot ───────────────────────────────────────────
        self.steps += 1
        self._print_step(self.steps, "🔧 Tool Call", "  [RetrievalAgent] get_market_summary()")
        snapshot = self._retrieval.fetch_market_snapshot()
        self.tool_results["get_market_summary"] = snapshot
        self.tool_calls.append(
            {"tool": "get_market_summary", "params": {}, "step": self.steps, "agent": "retrieval"}
        )
        self._print_step(
            self.steps, "📊 Tool Result",
            f"  Market snapshot fetched: {len(snapshot)} indices."
        )

        # ── Step 3: Equity research (only if user mentioned specific stocks) ──
        tickers = self._extract_tickers(user_query)
        # Filter out benchmark-like symbols already covered by market summary
        tickers = [
            t for t in tickers
            if t not in {"NSEI", "GSPC", "BSESN", "INR", "GCF"}
        ]
        if tickers:
            self.steps += 1
            self._print_step(
                self.steps, "🔧 Tool Call",
                f"  [ResearchAgent] get_stock_price() × {len(tickers)} tickers: {tickers}"
            )
            stock_data = self._research.run(tickers)
            self.tool_calls.extend([
                {**tc, "agent": "research"} for tc in self._research.tool_calls
            ])
            self._print_step(
                self.steps, "📊 Tool Result",
                f"  Stock data fetched for {len(stock_data)} tickers."
            )

        # ── Step 4: Risk profile ──────────────────────────────────────────────
        self.steps += 1
        self._print_step(
            self.steps, "🔧 Tool Call",
            f"  [AnalyticsAgent] calculate_risk(budget={profile['budget']}, "
            f"risk_level={profile['risk_level']}, age={profile['age']})"
        )
        risk_result = self._analytics.compute_risk_profile(
            budget=profile["budget"],
            risk_level=profile["risk_level"],
            age=profile["age"],
        )
        self.tool_results["calculate_risk"] = risk_result
        self.tool_calls.append({
            "tool": "calculate_risk",
            "params": profile,
            "step": self.steps,
            "agent": "analytics",
        })
        self._print_step(
            self.steps, "📊 Tool Result",
            f"  Risk score: {risk_result.get('risk_score', 'N/A')} | "
            f"Expected return: {risk_result.get('expected_annual_return', 'N/A')}"
        )

        # ── Step 5a: LLM picks instruments dynamically ────────────────────────
        self.steps += 1
        self._print_step(
            self.steps, "🤔 Thinking",
            f"  [ExecutionAgent] Asking LLM to select instruments for "
            f"{profile['risk_level']} risk | ₹{profile['budget']:,.0f} | {profile['goals']}"
        )
        custom_allocations = self._execution.pick_instruments_with_llm(
            budget=profile["budget"],
            risk_level=profile["risk_level"],
            goals=profile["goals"],
            risk_profile=risk_result,
            market_snapshot=snapshot,
        )
        if custom_allocations:
            self._print_step(
                self.steps, "📊 Tool Result",
                f"  LLM instrument picks ready → passing to suggest_portfolio."
            )
        else:
            self._print_step(
                self.steps, "⚠️ Fallback",
                "  LLM picker returned no valid picks — suggest_portfolio will use defaults."
            )

        # ── Step 5b: Portfolio construction ───────────────────────────────────
        self.steps += 1
        self._print_step(
            self.steps, "🔧 Tool Call",
            f"  [ExecutionAgent] suggest_portfolio(budget={profile['budget']}, "
            f"risk_level={profile['risk_level']}, "
            f"custom_allocations={'LLM picks' if custom_allocations else 'defaults'})"
        )
        portfolio_result = self._execution.build_portfolio(
            budget=profile["budget"],
            risk_level=profile["risk_level"],
            goals=profile["goals"],
            custom_allocations=custom_allocations,
        )
        self.tool_results["suggest_portfolio"] = portfolio_result
        self.tool_calls.append({
            "tool": "suggest_portfolio",
            "params": {"budget": profile["budget"], "risk_level": profile["risk_level"]},
            "step": self.steps,
            "agent": "execution",
        })
        self._print_step(
            self.steps, "📊 Tool Result",
            f"  Portfolio: {len(portfolio_result.get('instruments', []))} instruments."
        )

        # ── Step 6: Trend data for charts ─────────────────────────────────────
        self.steps += 1
        portfolio_symbols = self._execution.get_portfolio_symbols()
        self._print_step(
            self.steps, "🔧 Tool Call",
            f"  [RetrievalAgent] get_market_trends() for: "
            f"{['^NSEI', '^GSPC'] + portfolio_symbols}"
        )
        trend_result = self._retrieval.fetch_trends(portfolio_symbols)
        self.tool_results["get_market_trends"] = trend_result
        self.tool_calls.append({
            "tool": "get_market_trends",
            "params": {"symbols": portfolio_symbols},
            "step": self.steps,
            "agent": "retrieval",
        })
        self._print_step(
            self.steps, "📊 Tool Result",
            f"  Trend data: {len(trend_result.get('dates', []))} data points across "
            f"{len(trend_result.get('trends', {}))} symbols."
        )

        # ── Step 7: LLM synthesis — final report ──────────────────────────────
        self.steps += 1
        provider = active_provider()
        self._print_step(
            self.steps, "🤔 Thinking",
            f"  Synthesising final report via {provider} ({AGENT_MODEL})..."
        )

        llm_context = self._build_llm_context()
        synthesis_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": coordinator_system_prompt(TOOLS_SCHEMA, MAX_STEPS),
            },
            {
                "role": "user",
                "content": (
                    f"{user_query}\n\n"
                    f"All sub-agents have already gathered the following data:\n\n"
                    f"{llm_context}\n\n"
                    f"Based on this data, provide ONLY a final_answer JSON with a "
                    f"comprehensive, Markdown-formatted financial recommendation. "
                    f"Do NOT call any more tools."
                ),
            },
        ]

        final_answer: Optional[str] = None

        try:
            raw_response = call_llm(synthesis_messages)
            parsed = parse_json_response(raw_response)

            if parsed.get("action") == "final_answer":
                candidate = extract_final_answer(parsed.get("answer", raw_response))
                if not self._utility.is_garbage_answer(candidate):
                    final_answer = candidate
        except Exception as exc:
            self._print_step(self.steps, "❌ Error", f"  LLM synthesis failed: {exc}")

        # ── Step 8: Fallback report if LLM output is unusable ─────────────────
        if not final_answer:
            self._print_step(
                self.steps + 1,
                "⚠️ Fallback",
                "  LLM answer unusable — generating structured report from tool data.",
            )
            final_answer = self._utility.build_fallback_report(self.tool_results)

        self._print_step(self.steps, "✅ Final Answer", final_answer)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n\033[95m{'═'*60}\033[0m")
        print(f"  📋  Tools used: {[t['tool'] for t in self.tool_calls]}")
        print(f"  🔢  Total steps: {self.steps}")
        print(f"\033[95m{'═'*60}\033[0m\n")

        return final_answer
