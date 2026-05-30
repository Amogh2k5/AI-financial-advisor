<div align="center">
  <h1>💰 AI Wealth Advisor</h1>
  <p><strong>A Multi-Agent Financial Planning System — Powered by OpenRouter</strong></p>
</div>

---

## 🌟 Overview

**AI Wealth Advisor** is an intelligent, multi-agent financial planning application built to analyze market trends, ingest real-time data, and provide controlled, thoughtful financial advice. The system emphasizes the Indian market (Nifty 50, Sensex, Mutual Funds) while providing general financial analytical capabilities.

Leveraging **FastAPI**, **Server-Sent Events (SSE)**, and a highly controlled multi-agent loop, this project allows you to observe the AI's "Thinking" process live. It bridges the gap between raw market data and actionable, personalized financial insights with interactive charts and exportable PDF reports.

---

## 🏗️ Multi-Agent Architecture

Unlike simple chatbots, AI Wealth Advisor routes tasks across specialized sub-agents:

- 🧠 **`CoordinatorAgent`**: Orchestrates the entire flow and delegates to sub-agents.
- 📡 **`RetrievalAgent`**: Fetches live market snapshots and 30-day performance trends.
- 🔍 **`ResearchAgent`**: Performs specific stock/ticker lookups (e.g., via `yfinance`).
- 📈 **`AnalyticsAgent`**: Computes deterministic risk profiles based on user demographics and goals.
- 💼 **`ExecutionAgent`**: Builds portfolios, dynamically selecting instruments using an LLM.
- ⚙️ **`UtilityAgent`**: Ensures system resilience by generating deterministic fallback Markdown reports if the LLM synthesis fails.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.8+**
- **OpenRouter API Key**

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Use your OpenRouter API Key
OPENROUTER_API_KEY=your_key_here
MODEL_NAME=qwen2.5:7b
```

### 3. Setup the Environment & Run the Server
Open your primary terminal and run:

```bash
# Clone the repository and navigate into the root directory
cd AI_Wealth_Advisor

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Start the FastAPI Web Server
uvicorn backend.api:app --port 8000 --reload
```

### 4. Access the Dashboard
Once the server is running, the modern Glassmorphism dashboard is instantly available at:
👉 **[http://localhost:8000/](http://localhost:8000/)**

Interactive API documentation (Swagger UI) is available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧭 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Check if the LLM provider (OpenRouter) and Agents are ready. |
| `GET`  | `/api/market` | Instant live snapshot of global & Indian market benchmarks. |
| `POST` | `/api/analyze/stream` | **Real-time Multi-Agent Analysis** featuring live Server-Sent Events (SSE) streaming. |

---

## 🧠 Core Features

- ⚡ **Live Agent Feed (SSE):** Watch the AI's "Thinking" and reasoning process as it streams live to the dashboard's activity drawer.
- 📊 **Interactive Data Visualizations:** See where your money goes via dynamic Asset Allocation (Pie Charts) and compare 30-day performance trends (Line Charts).
- 🤖 **Resilient Multi-Agent Loop:** A robust system that fetches data, computes risk, builds portfolios, and always guarantees an output (even via fallback).
- 📉 **Export to PDF:** Instantly download your personalized financial strategy as a neat, formatted PDF.
- 🇮🇳 **Indian Market Focus:** Built-in heuristics configured specifically for analyzing Nifty 50, Sensex, and selecting high-performing Indian Mutual Funds.

---

*Built with ❤️ to simplify financial planning.*
