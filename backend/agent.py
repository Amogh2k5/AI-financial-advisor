"""
agent.py — The core agentic loop for the AI Wealth Advisor.

Architecture:
  1. Build a system prompt that includes all tool schemas.
  2. Send user query + conversation history to Ollama (qwen2.5:7b).
  3. Parse the model's JSON response:
       - If it contains {"action": "tool_name", "params": {...}} → run the tool.
       - If it contains {"action": "final_answer", "answer": "..."} → stop.
  4. Append tool results back to the conversation and loop.
  5. Hard-stop after MAX_STEPS to prevent infinite loops.
"""

import json
import re
import requests
import os
from dotenv import load_dotenv
from .tools import TOOLS, TOOLS_SCHEMA

# Load environment variables from .env file
load_dotenv(override=True)

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# API Keys and Models
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AGENT_MODEL = os.getenv("MODEL_NAME", "qwen2.5:7b") 

MAX_STEPS   = 10               # Increased steps for more complex analysis


# ─────────────────────────────────────────────
#  System Prompt Builder
# ─────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    Construct the system prompt that tells the LLM:
      - Its role
      - Available tools (with schemas)
      - Exact JSON formats to use for tool calls vs final answers
    """
    tool_descriptions = json.dumps(TOOLS_SCHEMA, indent=2)

    return f"""You are an expert AI Wealth Advisor helping Indian investors make smart financial decisions.

You have access to the following tools:
{tool_descriptions}

## How to respond

At each step, you MUST respond with ONLY a valid JSON object — no extra text, no markdown.

### To call a tool:
{{
  "action": "<tool_name>",
  "params": {{ "<param_name>": <value>, ... }},
  "reasoning": "<one sentence: why you are calling this tool>"
}}

### To give the final answer (after gathering enough data):
{{
  "action": "final_answer",
  "answer": "<your complete, well-structured financial recommendation in plain text>"
}}

## Rules
- Always start by understanding the user's budget, risk level, and goals.
- You MUST call `calculate_risk` and then `suggest_portfolio` BEFORE providing a `final_answer`. This is strictly required so that charts can be generated.
- Call tools in a logical order: market context → risk profile → portfolio suggestion.
- **IMPORTANT**: If you recommend specific stocks or funds in your portfolio, you MUST also call `get_market_trends(symbols="^NSEI,^GSPC,<symbol1>,<symbol2>...")` with those symbols (and indices) so the user can see a performance chart.
- Use the tool results to make your final recommendation concrete and data-driven.
- Do NOT call more than {MAX_STEPS} tools in total.
- Do NOT repeat a tool call with the same parameters.
- DO NOT write any Python scripts or code to calculate data manually. You must ONLY rely on the provided tools.
- Your final answer must be clear, structured, and actionable. Ensure you use proper Markdown formatting.
- Amounts should be in INR.
"""


# ─────────────────────────────────────────────
#  Ollama API Caller
# ─────────────────────────────────────────────

def call_openrouter(messages: list) -> str:
    """
    Send a list of chat messages to OpenRouter.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", # Optional
        "X-Title": "AI Wealth Advisor",
    }
    
    # OpenRouter doesn't support 'format': 'json' in the same way, 
    # but we force it in the prompt anyway.
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"❌ OpenRouter API error: {e}")

def call_llm(messages: list) -> str:
    """
    Routes the request to either OpenRouter or Ollama based on configuration.
    """
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_key_here":
        return call_openrouter(messages)
    else:
        return call_ollama(messages)

def call_ollama(messages: list) -> str:
    """
    Send a list of chat messages to Ollama and return the assistant's reply text.
    Uses the /api/chat endpoint with streaming disabled.
    """
    # Force use of local model if calling Ollama and AGENT_MODEL looks like a remote one
    model = AGENT_MODEL
    if "/" in AGENT_MODEL or "free" in AGENT_MODEL:
        model = "qwen2.5:7b" 
    
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "format":  "json",
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "❌ Cannot reach Ollama at localhost:11434. "
            "Make sure Ollama is running: run 'ollama serve' in a terminal."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("❌ Ollama took too long to respond. Try again.")
    except Exception as e:
        raise RuntimeError(f"❌ Ollama API error: {e}")


# ─────────────────────────────────────────────
#  JSON Parser (robust)
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """
    Extract and parse the first valid JSON object from the model's response.
    Handles:
      - Markdown code fences  (```json ... ```)
      - Literal newlines inside JSON string values (common LLM quirk)
      - Partial / truncated JSON via regex fallback
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # --- Attempt 1: standard parse of the first {...} block ---
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # --- Attempt 2: sanitise literal newlines inside string values ---
            # Replace unescaped newlines that sit inside double-quoted strings
            sanitised = re.sub(
                r'("(?:[^"\\]|\\.)*")',
                lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
                candidate,
            )
            try:
                return json.loads(sanitised)
            except json.JSONDecodeError:
                pass

    # --- Attempt 3: regex fallback — pull action + answer/params directly ---
    action_m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    answer_m = re.search(r'"answer"\s*:\s*"(.*)"\s*\}?\s*$', text, re.DOTALL | re.IGNORECASE)
    if action_m:
        result = {"action": action_m.group(1)}
        if answer_m:
            # Drop the trailing quote and optional brace if matched inside group 1
            ans = answer_m.group(1)
            if ans.endswith('"'): ans = ans[:-1]
            if ans.endswith('"\n}'): ans = ans[:-3]
            result["answer"] = ans.replace("\\n", "\n")
        return result

    # --- Attempt 4: last resort, return raw text as error ---
    return {"action": "error", "raw": text}


def extract_final_answer(value: str) -> str:
    """
    If the model returned the full JSON object as the final_answer string
    (e.g. when parse_json_response fell back to raw), strip the outer wrapper
    and return only the prose inside the 'answer' field.
    """
    stripped = value.strip()
    if stripped.startswith("{"):
        inner = parse_json_response(stripped)
        if inner.get("action") == "final_answer" and "answer" in inner:
            return inner["answer"]
            
        # Aggressive fallback: regex to extract string after "answer": "
        match = re.search(r'"answer"\s*:\s*"(.*)"\s*\}?\s*$', stripped, re.DOTALL | re.IGNORECASE)
        if match:
            ans = match.group(1)
            if ans.endswith('"'): ans = ans[:-1]
            if ans.endswith('"\n}'): ans = ans[:-3]
            ans = ans.replace('\\n', '\n').strip()
            # Clean up stray formatting quotes
            if ans.startswith('"') and ans.endswith('"'):
                ans = ans[1:-1]
            return ans
            
    # Clean up stray formatting quotes
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


# ─────────────────────────────────────────────
#  Tool Executor
# ─────────────────────────────────────────────

def _summarize_tool_result_for_llm(action: str, result: dict) -> str:
    """
    Return a compact summary of a tool result to send to the LLM.
    This avoids overwhelming small local models with thousands of data points.
    """
    if action == "get_market_trends":
        # Never send raw trend data to LLM — it's only needed for the chart
        trends_summary = {}
        for sym, vals in result.get("trends", {}).items():
            if isinstance(vals, list) and vals:
                trends_summary[sym] = {
                    "start_pct": vals[0],
                    "end_pct": vals[-1],
                    "total_return_pct": round(vals[-1] - vals[0], 2),
                }
        return json.dumps({
            "status": "Trend data fetched successfully for charts.",
            "summary": trends_summary,
            "note": "Full trend data will be shown in the performance chart."
        }, indent=2)

    if action == "get_market_summary":
        # Trim to just name, price, percent_change
        compact = {}
        for sym, data in result.items():
            if isinstance(data, dict):
                compact[sym] = {
                    "name": data.get("name"),
                    "price": data.get("price"),
                    "change_pct": data.get("percent_change")
                }
        return json.dumps(compact, indent=2, default=str)

    # For all other tools, send the full result
    return json.dumps(result, indent=2, default=str)


def execute_tool(action: str, params: dict) -> tuple:
    """
    Look up the tool function by name, call it with the provided params.
    Returns (full_result_dict, llm_summary_str).
    """
    if action not in TOOLS:
        err = {"error": f"Unknown tool: '{action}'"}
        return err, json.dumps(err)

    tool_fn = TOOLS[action]

    try:
        result = tool_fn(**params)
        llm_str = _summarize_tool_result_for_llm(action, result)
        return result, llm_str
    except TypeError as e:
        err = {"error": f"Wrong parameters for '{action}': {e}"}
        return err, json.dumps(err)
    except Exception as e:
        err = {"error": f"Tool '{action}' failed: {e}"}
        return err, json.dumps(err)


# ─────────────────────────────────────────────
#  Main Agent Class
# ─────────────────────────────────────────────

class WealthAdvisorAgent:
    """
    Controlled agentic loop:
      user query → LLM → tool call (repeated up to MAX_STEPS) → final answer
    """

    def __init__(self, event_queue=None):
        self.messages:     list = []
        self.steps:        int  = 0
        self.tool_calls:   list = []   # Log of every tool call made
        self.tool_results: dict = {}   # Captured tool results for UI
        self.event_queue         = event_queue   # queue.Queue | None — for SSE streaming

    # ── Reset state between sessions ──────────────────────────────────────────

    def reset(self):
        self.messages   = []
        self.steps      = 0
        self.tool_calls = []
        self.tool_results = {}
        # NOTE: event_queue is intentionally NOT reset — it belongs to the caller

    # ── Event emitter — works for both CLI (print) and API (queue) ─────────────

    # Label → short event_type string used by the frontend
    _LABEL_TO_TYPE = {
        "🤔 Thinking":    "thinking",
        "🔧 Tool Call":   "tool_call",
        "📊 Tool Result": "tool_result",
        "✅ Final Answer": "final_answer",
        "❌ Error":       "error",
    }

    def _print_step(self, step: int, label: str, content: str):
        """
        Print the step to the console (CLI mode) AND push a structured event
        to the event_queue if one was supplied (API / SSE mode).
        """
        # ── Console output (always shown) ────────────────────────────────────
        colour_reset = "\033[0m"
        colours = {
            "🤔 Thinking":    "\033[94m",
            "🔧 Tool Call":   "\033[93m",
            "📊 Tool Result": "\033[92m",
            "✅ Final Answer": "\033[96m",
            "❌ Error":       "\033[91m",
        }
        colour = colours.get(label, "")
        print(f"\n{colour}{'─'*60}")
        print(f"  Step {step} | {label}")
        print(f"{'─'*60}{colour_reset}")
        display = content if len(content) < 800 else content[:800] + "\n  ... [truncated]"
        print(display)

        # ── SSE queue push (API mode only) ────────────────────────────────────
        if self.event_queue is not None:
            self.event_queue.put({
                "type":    self._LABEL_TO_TYPE.get(label, "info"),
                "step":    step,
                "label":   label,
                "content": content,
            })

    # ── Core agent loop ───────────────────────────────────────────────────────

    def run(self, user_query: str) -> str:
        """
        Execute the full agent loop for a given user query.
        Returns the final answer string.
        """
        self.reset()

        # Initialise conversation with system prompt + user message
        self.messages = [
            {"role": "system",  "content": build_system_prompt()},
            {"role": "user",    "content": user_query},
        ]

        print(f"\n\033[95m{'═'*60}")
        print(f"  🤖  AI Wealth Advisor — Agent Started")
        print(f"{'═'*60}\033[0m")
        print(f"  Query: {user_query}")

        final_answer = None

        while self.steps < MAX_STEPS:
            self.steps += 1
            provider = "OpenRouter" if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_key_here" else "Ollama"
            self._print_step(self.steps, "🤔 Thinking", f"  Sending request to {provider} ({AGENT_MODEL})...")

            # ── Get LLM response ─────────────────────────────────────────────
            raw_response = call_llm(self.messages)
            parsed       = parse_json_response(raw_response)

            action = parsed.get("action", "error")

            # ── Final Answer ─────────────────────────────────────────────────
            if action == "final_answer":
                candidate = extract_final_answer(parsed.get("answer", raw_response))
                # Reject if it looks like a tool-call JSON was passed as the answer
                if self._is_garbage_answer(candidate):
                    self._print_step(self.steps, "❌ Error", "  LLM returned a tool call as the final answer. Will use fallback.")
                    final_answer = None  # Force fallback
                else:
                    final_answer = candidate
                self._print_step(self.steps, "✅ Final Answer", final_answer or "[using fallback]")
                break

            # ── Error / Unparseable response ─────────────────────────────────
            if action == "error":
                self._print_step(
                    self.steps, "❌ Error",
                    f"  Could not parse LLM response:\n  {parsed.get('raw', raw_response)[:300]}"
                )
                # Give the model a nudge and try once more
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON. "
                        "Please respond with ONLY a valid JSON object using the format specified."
                    ),
                })
                continue

            # ── Tool Call ────────────────────────────────────────────────────
            params    = parsed.get("params", {})
            reasoning = parsed.get("reasoning", "")

            self._print_step(
                self.steps, "🔧 Tool Call",
                f"  Tool:      {action}\n"
                f"  Params:    {json.dumps(params)}\n"
                f"  Reasoning: {reasoning}"
            )

            # Log the tool call
            self.tool_calls.append({"tool": action, "params": params, "step": self.steps})

            # Execute the tool
            full_result, llm_summary = execute_tool(action, params)

            # Store full structured data for charts/frontend
            if action in ["suggest_portfolio", "get_market_trends", "calculate_risk"]:
                if isinstance(full_result, dict) and "error" not in full_result:
                    self.tool_results[action] = full_result

            self._print_step(self.steps, "📊 Tool Result", llm_summary)

            # Append only the compact summary to conversation (not raw data)
            self.messages.append({"role": "assistant", "content": raw_response})
            self.messages.append({
                "role": "user",
                "content": (
                    f"Tool '{action}' returned the following result:\n"
                    f"{llm_summary}\n\n"
                    f"Continue your analysis. If you have enough information, "
                    f"provide the final_answer. Otherwise, call the next tool."
                ),
            })

        # ── AUTO-CALL suggest_portfolio if LLM skipped it ────────────────────
        if "suggest_portfolio" not in self.tool_results:
            print("\n\033[94m🔄 Auto-calling suggest_portfolio (LLM skipped it)...\033[0m")
            try:
                risk_data = self.tool_results.get("calculate_risk", {})
                budget = risk_data.get("budget_inr", 100000)
                risk_level = risk_data.get("risk_level", "medium")
                from .tools import suggest_portfolio
                portfolio_data = suggest_portfolio(budget=budget, risk_level=risk_level)
                self.tool_results["suggest_portfolio"] = portfolio_data
                self.tool_calls.append({"tool": "suggest_portfolio", "params": {"budget": budget, "risk_level": risk_level}, "step": self.steps})
            except Exception as e:
                print(f"⚠️ Failed to auto-call suggest_portfolio: {e}")

        # ── AUTO-ENRICH TRENDS (Sync Charts) ──────────────────────────────────
        if "suggest_portfolio" in self.tool_results and "get_market_trends" not in self.tool_results:
            try:
                portfolio = self.tool_results["suggest_portfolio"]
                portfolio_symbols = [i.get("symbol") for i in portfolio.get("instruments", []) if i.get("symbol") and i.get("symbol") not in ["CASH", "GOLD"]]
                all_symbols = ["^NSEI", "^GSPC"] + portfolio_symbols
                all_symbols = list(dict.fromkeys(all_symbols))  # unique, preserve order
                print(f"\n\033[94m🔄 Auto-fetching performance trends for: {all_symbols}\033[0m")
                from .tools import get_market_trends
                trend_data = get_market_trends(all_symbols)
                self.tool_results["get_market_trends"] = trend_data
            except Exception as e:
                print(f"⚠️ Failed to auto-enrich trends: {e}")

        # ── Validate and clean final answer ──────────────────────────────────
        if final_answer and self._is_garbage_answer(final_answer):
            print("\n\033[93m⚠️  Final answer is garbage JSON/tool call. Using fallback.\033[0m")
            final_answer = None

        if final_answer is None:
            print(f"\n\033[93m⚠️  No valid final answer. Generating structured report...\033[0m")
            final_answer = self._build_fallback_answer()
            self._print_step(self.steps + 1, "✅ Final Answer", final_answer)

        print(f"\n\033[95m{'═'*60}\033[0m")
        print(f"  📋  Tools used: {[t['tool'] for t in self.tool_calls]}")
        print(f"  🔢  Total steps: {self.steps}")
        print(f"\033[95m{'═'*60}\033[0m\n")

        return final_answer

    def _is_garbage_answer(self, text: str) -> bool:
        """Return True if the 'answer' looks like a tool call JSON or raw code block."""
        if not text:
            return True
        stripped = text.strip()
        # Looks like a JSON tool call
        if '{"action"' in stripped.replace(' ', '') or '{"tool"' in stripped.replace(' ', ''):
            return True
        # Looks like Python code
        if 'import numpy' in stripped or 'import pandas' in stripped or stripped.startswith('```python'):
            return True
        # Too short to be a real financial report (less than 80 chars)
        if len(stripped) < 80:
            return True
        return False

    def _build_fallback_answer(self) -> str:
        """Generate a structured markdown report from collected tool results."""
        parts = ["# 📊 AI Wealth Advisor — Financial Strategy Report\n"]
        parts.append("> ⚠️ *This plan was generated by the AI Wealth Advisor. Always consult a SEBI-registered advisor before investing.*\n")

        risk = self.tool_results.get("calculate_risk", {})
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
                parts.append(f"- **Allocation:** {alloc.get('equity', 0)}% Equity | {alloc.get('bonds', 0)}% Bonds | {alloc.get('gold', 0)}% Gold | {alloc.get('cash', 0)}% Cash")
            parts.append("")

        portfolio = self.tool_results.get("suggest_portfolio", {})
        if portfolio:
            instruments = portfolio.get("instruments", [])
            budget = portfolio.get("total_budget_inr", 0)
            parts.append("## 💼 Recommended Portfolio")
            parts.append(f"**Total Budget: ₹{budget:,.0f}** | Risk: {portfolio.get('risk_level', 'N/A').capitalize()}\n")
            parts.append("| # | Investment | Type | Allocation | Amount (INR) | Why |")
            parts.append("|---|-----------|------|-----------|--------------|-----|")
            for i, inst in enumerate(instruments, 1):
                parts.append(f"| {i} | **{inst.get('name', 'N/A')}** | {inst.get('type', '')} | {inst.get('weight_pct', 0):.1f}% | ₹{inst.get('amount_inr', 0):,.0f} | {inst.get('why', '')} |")
            parts.append("")

        parts.append("## 📋 Key Recommendations")
        risk_level = risk.get("risk_level", "medium") if risk else "medium"
        if risk_level == "low":
            parts.append("- 🛡️ **Capital Preservation First** — Focus on FDs, bonds, and liquid mutual funds.")
            parts.append("- 💛 **Gold as Hedge** — Allocate to Sovereign Gold Bonds for inflation protection.")
            parts.append("- 📅 **SIP Strategy** — Start a monthly SIP in a large-cap or index fund.")
        elif risk_level == "medium":
            parts.append("- ⚖️ **Balanced Approach** — Mix of equity mutual funds and debt instruments.")
            parts.append("- 📈 **Nifty 50 Index Fund** — Low-cost index funds for core equity exposure.")
            parts.append("- 🏦 **HDFC / SBI Blue Chips** — Add quality banking stocks for stability.")
            parts.append("- 💛 **10-20% Gold** — As a portfolio hedge against market volatility.")
        else:
            parts.append("- 🚀 **Aggressive Growth** — High allocation to equity, especially mid & small caps.")
            parts.append("- 🌐 **Global Exposure** — Consider US tech stocks (NVDA, MSFT) via international funds.")
            parts.append("- 📊 **Momentum Stocks** — Rotate into high-momentum sectors like IT and pharma.")
            parts.append("- ⚠️ **Set Stop-Losses** — Given high volatility, always protect your downside.")
        parts.append("")
        parts.append("## 🔔 Disclaimer")
        parts.append("*This AI-generated report is for educational purposes only. Past performance is not indicative of future results. Please consult a SEBI-registered financial advisor before making any investment decisions.*")

        return "\n".join(parts)
