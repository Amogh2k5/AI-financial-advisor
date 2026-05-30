"""
main.py — Entry point for the AI Wealth Advisor.

Collects user profile (budget, risk level, goals, age),
builds a structured query, and hands it to the WealthAdvisorAgent.
"""

import sys
from .agents import CoordinatorAgent  # Multi-agent orchestrator


# ─────────────────────────────────────────────
#  Colour helpers (Windows-safe)
# ─────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code."""
    return f"\033[{code}m{text}\033[0m"

def banner():
    print(_c("""
╔══════════════════════════════════════════════════════════════╗
║          💰  AI WEALTH ADVISOR  — Powered by Qwen 2.5        ║
║               Agentic Financial Intelligence System          ║
╚══════════════════════════════════════════════════════════════╝
""", "96"))


# ─────────────────────────────────────────────
#  User input collector
# ─────────────────────────────────────────────

def collect_user_profile() -> dict:
    """
    Interactively gather the investor's profile from stdin.
    Returns a dict with: name, age, budget, risk_level, goals, stocks_of_interest.
    """
    print(_c("\n  Please answer a few quick questions to get started:\n", "93"))

    # Name
    name = input("  👤 Your name: ").strip() or "Investor"

    # Age — validate it's a number
    while True:
        try:
            age = int(input("  🎂 Your age: ").strip())
            if 18 <= age <= 100:
                break
            print("  ⚠️  Please enter an age between 18 and 100.")
        except ValueError:
            print("  ⚠️  Please enter a valid number.")

    # Budget — validate numeric
    while True:
        raw = input("  💵 Investment budget (INR): ₹").strip().replace(",", "")
        try:
            budget = float(raw)
            if budget > 0:
                break
            print("  ⚠️  Budget must be greater than zero.")
        except ValueError:
            print("  ⚠️  Please enter a numeric amount (e.g. 50000).")

    # Risk level
    print("\n  Risk levels:")
    print("    [1] Low    — Preserve capital, steady returns")
    print("    [2] Medium — Balanced growth with moderate risk")
    print("    [3] High   — Aggressive growth, higher volatility")
    while True:
        choice = input("\n  📊 Choose risk level (1/2/3): ").strip()
        risk_map = {"1": "low", "2": "medium", "3": "high",
                    "low": "low", "medium": "medium", "high": "high"}
        if choice.lower() in risk_map:
            risk_level = risk_map[choice.lower()]
            break
        print("  ⚠️  Please enter 1, 2, or 3.")

    # Investment goals
    print("\n  Common goals: retirement, house, education, travel, wealth creation")
    goals = input("  🎯 Your investment goals: ").strip() or "long-term wealth creation"

    # Optional: specific stocks to look up
    stocks_raw = input(
        "\n  📈 Any specific stocks to analyse? (comma-separated tickers, or press Enter to skip): "
    ).strip()
    stocks = [s.strip().upper() for s in stocks_raw.split(",") if s.strip()] if stocks_raw else []

    return {
        "name":              name,
        "age":               age,
        "budget":            budget,
        "risk_level":        risk_level,
        "goals":             goals,
        "stocks_of_interest": stocks,
    }


# ─────────────────────────────────────────────
#  Query builder
# ─────────────────────────────────────────────

def build_query(profile: dict) -> str:
    """
    Convert a structured profile dict into a natural-language query
    that the agent can process effectively.
    """
    budget_str = f"₹{profile['budget']:,.0f}"
    stocks_str = (
        f"Also analyse these specific stocks for me: {', '.join(profile['stocks_of_interest'])}."
        if profile["stocks_of_interest"]
        else ""
    )

    return (
        f"I am {profile['name']}, {profile['age']} years old. "
        f"I have {budget_str} to invest. "
        f"My risk tolerance is {profile['risk_level']}. "
        f"My financial goals are: {profile['goals']}. "
        f"{stocks_str} "
        f"Please: "
        f"1) Check the current market conditions, "
        f"2) Calculate my risk profile and suitable asset allocation, "
        f"3) Suggest a detailed portfolio with specific instruments and amounts. "
        f"Give me a complete, actionable financial plan."
    )


# ─────────────────────────────────────────────
#  Output formatter — final recommendation
# ─────────────────────────────────────────────

def display_recommendation(name: str, answer: str):
    print(_c(f"\n{'═'*62}", "96"))
    print(_c(f"  📋  FINANCIAL RECOMMENDATION FOR {name.upper()}", "96"))
    print(_c(f"{'═'*62}\n", "96"))
    print(answer)
    print(_c(f"\n{'═'*62}", "96"))
    print(_c(
        "  ⚠️  DISCLAIMER: This is an AI-generated educational plan.\n"
        "      Always consult a SEBI-registered financial advisor\n"
        "      before making real investment decisions.",
        "33"
    ))
    print(_c(f"{'═'*62}\n", "96"))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    # Enable ANSI colours on Windows
    import os
    os.system("")   # Activates VT100 mode on Windows console

    banner()

    agent = CoordinatorAgent()

    while True:
        try:
            # Step 1: collect investor profile
            profile = collect_user_profile()

            # Step 2: build structured query
            query = build_query(profile)

            print(_c(f"\n  🚀 Starting analysis for {profile['name']}...", "92"))

            # Step 3: run the agent
            final_answer = agent.run(query)

            # Step 4: display formatted recommendation
            display_recommendation(profile["name"], final_answer)

        except ConnectionError as e:
            print(_c(f"\n{e}", "91"))
            print(_c("  → Start Ollama with: ollama serve", "93"))
            sys.exit(1)

        except KeyboardInterrupt:
            print(_c("\n\n  👋 Goodbye! Invest wisely.\n", "96"))
            sys.exit(0)

        except Exception as e:
            print(_c(f"\n  ❌ Unexpected error: {e}", "91"))

        # Ask if they want another session
        print()
        again = input("  🔄 Analyse another profile? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print(_c("\n  👋 Thank you for using AI Wealth Advisor. Invest wisely!\n", "96"))
            break


if __name__ == "__main__":
    main()
