
import sys
import os
from dotenv import load_dotenv

# Add the project root to sys.path so we can import 'backend'
sys.path.append(os.getcwd())

try:
    from backend.agent import WealthAdvisorAgent
    from backend.tools import get_market_summary, get_market_trends, calculate_risk, suggest_portfolio
    print("[OK] Imports successful")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test tools
try:
    summary = get_market_summary()
    print(f"[OK] get_market_summary: {len(summary)} items")
    
    risk = calculate_risk(100000, "medium", 30)
    print(f"[OK] calculate_risk: {risk['risk_level']}")
    
    portfolio = suggest_portfolio(100000, "medium")
    print(f"[OK] suggest_portfolio: {len(portfolio['instruments'])} instruments")
    
    trends = get_market_trends("^NSEI,^GSPC")
    print(f"[OK] get_market_trends: {len(trends['dates'])} dates")
except Exception as e:
    print(f"[ERROR] Tool test failed: {e}")

# Test Agent initialization
try:
    agent = WealthAdvisorAgent()
    print("[OK] WealthAdvisorAgent initialized")
except Exception as e:
    print(f"[ERROR] Agent init failed: {e}")
