"""
utility_agent.py — Shared LLM infrastructure for the AI Wealth Advisor.

Responsibilities:
  - call_llm()            : Routes requests to OpenRouter or Ollama.
  - call_openrouter()     : OpenRouter API caller (primary provider).
  - call_ollama()         : Ollama API caller (local fallback).
  - parse_json_response() : Robust JSON extractor for LLM output.
  - extract_final_answer(): Unwrap nested JSON final_answer values.
  - execute_tool()        : Dispatch tool calls with error handling.
  - UtilityAgent          : Generates structured fallback reports.

All other agents import from this module — it is the shared foundation.
"""

import json
import math
import re
from typing import Any, Dict, Tuple

import requests

from ..tools import TOOLS
from .config import (
    AGENT_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    is_openrouter_configured,
)
from .prompts import utility_fallback_prompt


# ─────────────────────────────────────────────
#  OpenRouter API Caller  (primary provider)
# ─────────────────────────────────────────────

def call_openrouter(messages: list) -> str:
    """
    Send a list of chat messages to OpenRouter and return the assistant's reply.

    Args:
        messages: A list of dicts with 'role' and 'content' keys
                  (standard OpenAI-compatible format).

    Returns:
        The raw text content of the assistant's response.

    Raises:
        RuntimeError: If the API call fails for any reason.
    """
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Wealth Advisor",
    }

    payload: Dict[str, Any] = {
        "model": AGENT_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"❌ OpenRouter API error: {exc}") from exc


# ─────────────────────────────────────────────
#  Ollama API Caller  (local fallback)
# ─────────────────────────────────────────────

def call_ollama(messages: list) -> str:
    """
    Send a list of chat messages to a local Ollama instance.

    Falls back to 'qwen2.5:7b' if the configured model looks like a remote
    OpenRouter model (contains '/' or 'free').

    Args:
        messages: Standard OpenAI-compatible message list.

    Returns:
        The raw text content of the assistant's response.

    Raises:
        ConnectionError: If Ollama is not running at localhost:11434.
        TimeoutError:    If Ollama takes too long to respond.
        RuntimeError:    For all other Ollama-related errors.
    """
    model = AGENT_MODEL
    if "/" in AGENT_MODEL or "free" in AGENT_MODEL:
        model = "qwen2.5:7b"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            "❌ Cannot reach Ollama at localhost:11434. "
            "Make sure Ollama is running: run 'ollama serve' in a terminal."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError("❌ Ollama took too long to respond. Try again.") from exc
    except Exception as exc:
        raise RuntimeError(f"❌ Ollama API error: {exc}") from exc


# ─────────────────────────────────────────────
#  Unified LLM Router
# ─────────────────────────────────────────────

def call_llm(messages: list) -> str:
    """
    Route the LLM request to OpenRouter (if configured) or Ollama (fallback).

    Args:
        messages: Standard OpenAI-compatible message list.

    Returns:
        The assistant's raw text response.
    """
    if is_openrouter_configured():
        return call_openrouter(messages)
    return call_ollama(messages)


# ─────────────────────────────────────────────
#  JSON Response Parser
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first valid JSON object from an LLM response.

    Handles:
      - Markdown code fences  (```json ... ```)
      - Literal unescaped newlines inside JSON string values
      - Partial / truncated JSON via regex fallback
      - Completely unparseable responses (returns error sentinel)

    Args:
        text: Raw string output from the LLM.

    Returns:
        A dict with at least an 'action' key.
        On failure: {"action": "error", "raw": <original_text>}
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Attempt 1: parse first {...} block directly
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt 2: sanitise literal newlines inside string values
            sanitised = re.sub(
                r'("(?:[^"\\]|\\.)*")',
                lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
                candidate,
            )
            try:
                return json.loads(sanitised)
            except json.JSONDecodeError:
                pass

    # Attempt 3: regex fallback — pull action + answer/params directly
    action_m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    answer_m = re.search(r'"answer"\s*:\s*"(.*)"?\s*\}?\s*$', text, re.DOTALL | re.IGNORECASE)
    if action_m:
        result: Dict[str, Any] = {"action": action_m.group(1)}
        if answer_m:
            ans = answer_m.group(1)
            if ans.endswith('"'):
                ans = ans[:-1]
            if ans.endswith('"\n}'):
                ans = ans[:-3]
            result["answer"] = ans.replace("\\n", "\n")
        return result

    # Attempt 4: last resort — return raw text as error
    return {"action": "error", "raw": text}


# ─────────────────────────────────────────────
#  Final Answer Extractor
# ─────────────────────────────────────────────

def extract_final_answer(value: str) -> str:
    """
    Unwrap nested JSON that a model sometimes returns as the 'answer' value.

    If the LLM embeds a full JSON object (with 'action':'final_answer') inside
    the answer string, this function extracts only the inner prose.

    Args:
        value: The raw string from parsed["answer"].

    Returns:
        Clean, unwrapped answer string.
    """
    stripped = value.strip()

    if stripped.startswith("{"):
        inner = parse_json_response(stripped)
        if inner.get("action") == "final_answer" and "answer" in inner:
            return inner["answer"]

        # Aggressive regex fallback
        match = re.search(
            r'"answer"\s*:\s*"(.*)"?\s*\}?\s*$', stripped, re.DOTALL | re.IGNORECASE
        )
        if match:
            ans = match.group(1)
            if ans.endswith('"'):
                ans = ans[:-1]
            if ans.endswith('"\n}'):
                ans = ans[:-3]
            ans = ans.replace("\\n", "\n").strip()
            if ans.startswith('"') and ans.endswith('"'):
                ans = ans[1:-1]
            return ans

    # Clean stray surrounding quotes
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


# ─────────────────────────────────────────────
#  Tool Result Summariser  (for LLM context)
# ─────────────────────────────────────────────

def summarise_tool_result_for_llm(action: str, result: Dict[str, Any]) -> str:
    """
    Return a compact summary of a tool result to inject into the LLM conversation.

    Large datasets (e.g., 30-day trend arrays) are condensed to avoid overwhelming
    smaller models with thousands of data points.

    Args:
        action: The tool name that produced this result.
        result: The raw dict returned by the tool function.

    Returns:
        A compact JSON string suitable for LLM consumption.
    """
    if action == "get_market_trends":
        trends_summary: Dict[str, Any] = {}
        for sym, vals in result.get("trends", {}).items():
            if isinstance(vals, list) and vals:
                trends_summary[sym] = {
                    "start_pct": vals[0],
                    "end_pct": vals[-1],
                    "total_return_pct": round(vals[-1] - vals[0], 2),
                }
        return json.dumps(
            {
                "status": "Trend data fetched successfully for charts.",
                "summary": trends_summary,
                "note": "Full trend data will be shown in the performance chart.",
            },
            indent=2,
        )

    if action == "get_market_summary":
        compact: Dict[str, Any] = {}
        for sym, data in result.items():
            if isinstance(data, dict):
                compact[sym] = {
                    "name": data.get("name"),
                    "price": data.get("price"),
                    "change_pct": data.get("percent_change"),
                }
        return json.dumps(compact, indent=2, default=str)

    # All other tools: send the full result
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────
#  Tool Executor
# ─────────────────────────────────────────────

def execute_tool(action: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Look up a tool by name and execute it with the provided parameters.

    Args:
        action: The tool name (must be a key in TOOLS).
        params: A dict of keyword arguments to pass to the tool function.

    Returns:
        A tuple of (full_result_dict, llm_summary_str).
        On error: returns an error dict and its JSON representation.
    """
    if action not in TOOLS:
        err: Dict[str, Any] = {"error": f"Unknown tool: '{action}'"}
        return err, json.dumps(err)

    tool_fn = TOOLS[action]

    try:
        result: Dict[str, Any] = tool_fn(**params)
        llm_str = summarise_tool_result_for_llm(action, result)
        return result, llm_str
    except TypeError as exc:
        err = {"error": f"Wrong parameters for '{action}': {exc}"}
        return err, json.dumps(err)
    except Exception as exc:
        err = {"error": f"Tool '{action}' failed: {exc}"}
        return err, json.dumps(err)


# ─────────────────────────────────────────────
#  Utility Agent — Fallback Report Builder
# ─────────────────────────────────────────────

class UtilityAgent:
    """
    Utility / helper agent that generates a structured Markdown report from
    collected tool results when the coordinator's LLM answer is unusable.

    This agent does NOT make LLM calls — it synthesises data deterministically.
    It can optionally call the LLM with a lightweight prompt for richer prose.
    """

    def build_fallback_report(self, tool_results: Dict[str, Any]) -> str:
        """
        Generate a structured Markdown financial report from raw tool results.

        Args:
            tool_results: Dict mapping tool names to their result dicts,
                          e.g. {"calculate_risk": {...}, "suggest_portfolio": {...}}.

        Returns:
            A Markdown-formatted financial strategy report string.
        """
        parts: list = [
            "# 📊 AI Wealth Advisor — Financial Strategy Report\n",
            "> ⚠️ *This plan was generated by the AI Wealth Advisor. "
            "Always consult a SEBI-registered advisor before investing.*\n",
        ]

        risk = tool_results.get("calculate_risk", {})
        if risk:
            alloc = risk.get("allocation", {})
            parts.append("## 🧮 Your Risk Profile")
            parts.append(f"- **Risk Level:** {risk.get('risk_level', 'N/A').capitalize()}")
            parts.append(f"- **Budget:** ₹{risk.get('budget_inr', 0):,.0f}")
            parts.append(f"- **Age:** {risk.get('age', 'N/A')} years")
            parts.append(f"- **Expected Annual Return:** {risk.get('expected_annual_return', 'N/A')}")
            parts.append(f"- **Max Drawdown Tolerance:** {risk.get('max_drawdown_pct', 'N/A')}%")
            parts.append(f"- **Investment Horizon:** {risk.get('investment_horizon', 'N/A')}")
            if alloc:
                parts.append(
                    f"- **Allocation:** {alloc.get('equity', 0)}% Equity | "
                    f"{alloc.get('bonds', 0)}% Bonds | "
                    f"{alloc.get('gold', 0)}% Gold | "
                    f"{alloc.get('cash', 0)}% Cash"
                )
            parts.append("")

        portfolio = tool_results.get("suggest_portfolio", {})
        if portfolio:
            instruments = portfolio.get("instruments", [])
            budget = portfolio.get("total_budget_inr", 0)
            parts.append("## 💼 Recommended Portfolio")
            parts.append(
                f"**Total Budget: ₹{budget:,.0f}** | "
                f"Risk: {portfolio.get('risk_level', 'N/A').capitalize()}\n"
            )
            parts.append("| # | Investment | Type | Allocation | Amount (INR) | Why |")
            parts.append("|---|-----------|------|-----------|--------------|-----|")
            for i, inst in enumerate(instruments, 1):
                parts.append(
                    f"| {i} | **{inst.get('name', 'N/A')}** | {inst.get('type', '')} | "
                    f"{inst.get('weight_pct', 0):.1f}% | ₹{inst.get('amount_inr', 0):,.0f} | "
                    f"{inst.get('why', '')} |"
                )
            parts.append("")

        parts.append("## 📋 Key Recommendations")
        risk_level = risk.get("risk_level", "medium") if risk else "medium"
        if risk_level == "low":
            parts += [
                "- 🛡️ **Capital Preservation First** — Focus on FDs, bonds, and liquid mutual funds.",
                "- 💛 **Gold as Hedge** — Allocate to Sovereign Gold Bonds for inflation protection.",
                "- 📅 **SIP Strategy** — Start a monthly SIP in a large-cap or index fund.",
            ]
        elif risk_level == "medium":
            parts += [
                "- ⚖️ **Balanced Approach** — Mix of equity mutual funds and debt instruments.",
                "- 📈 **Nifty 50 Index Fund** — Low-cost index funds for core equity exposure.",
                "- 🏦 **HDFC / SBI Blue Chips** — Add quality banking stocks for stability.",
                "- 💛 **10-20% Gold** — As a portfolio hedge against market volatility.",
            ]
        else:
            parts += [
                "- 🚀 **Aggressive Growth** — High allocation to equity, especially mid & small caps.",
                "- 🌐 **Global Exposure** — Consider US tech stocks (NVDA, MSFT) via international funds.",
                "- 📊 **Momentum Stocks** — Rotate into high-momentum sectors like IT and pharma.",
                "- ⚠️ **Set Stop-Losses** — Given high volatility, always protect your downside.",
            ]

        parts.append("")
        parts.append("## 🔔 Disclaimer")
        parts.append(
            "*This AI-generated report is for educational purposes only. "
            "Past performance is not indicative of future results. "
            "Please consult a SEBI-registered financial advisor before making any investment decisions.*"
        )

        return "\n".join(parts)

    def is_garbage_answer(self, text: str) -> bool:
        """
        Return True if the answer string looks like a tool-call JSON or raw code,
        indicating the LLM failed to produce a proper natural-language response.

        Args:
            text: The candidate final answer string.

        Returns:
            True if the answer should be discarded, False otherwise.
        """
        if not text:
            return True
        stripped = text.strip()
        if '{"action"' in stripped.replace(" ", "") or '{"tool"' in stripped.replace(" ", ""):
            return True
        if (
            "import numpy" in stripped
            or "import pandas" in stripped
            or stripped.startswith("```python")
        ):
            return True
        # Too short to be a real financial report
        if len(stripped) < 80:
            return True
        return False
