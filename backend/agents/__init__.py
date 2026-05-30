"""
agents/ — Multi-agent package for the AI Wealth Advisor.

Sub-agents:
  - research_agent   : Market research & stock-price lookup
  - retrieval_agent  : Historical trends & market snapshots
  - analytics_agent  : Risk profiling & portfolio analytics
  - execution_agent  : Portfolio construction & allocation
  - utility_agent    : Shared LLM helpers (call_llm, parse_json, execute_tool)

Orchestration:
  - coordinator_agent : Routes queries to the appropriate sub-agents and
                        assembles the final financial recommendation.
"""

from .coordinator_agent import CoordinatorAgent

__all__ = ["CoordinatorAgent"]
