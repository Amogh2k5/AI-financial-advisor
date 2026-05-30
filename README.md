<div align="center">
  <h1>💰 AI Wealth Advisor</h1>
  <p><strong>An Agentic AI Financial Planning System — Powered by Qwen 2.5</strong></p>
</div>

---

## 🌟 Overview

**AI Wealth Advisor** is an intelligent, agent-driven financial planning application built to analyze market trends, ingest real-time data, and provide controlled, thoughtful financial advice. The system emphasizes the Indian market (Nifty 50, Sensex, Mutual Funds) while providing general financial analytical capabilities.

Leveraging **FastAPI**, **Server-Sent Events (SSE)**, and a highly controlled agentic loop, this project allows you to observe the AI's "Thinking" process live, bridging the gap between raw data and actionable financial insights.

---

## 🏗️ Project Architecture

```text
AI_Wealth_Advisor/
├── backend/            <-- Python Agent & API Layer
│   ├── api.py          ← FastAPI Web Server (Endpoints & Streaming)
│   ├── agent.py        ← Controlled Agentic Loop (Core Intelligence)
│   ├── tools.py        ← Financial Toolset (Data Fetching & Calculation)
│   └── main.py         ← CLI Entry Point (Optional)
├── frontend/           <-- Interactive Web Dashboard
│   ├── index.html      ← UI Structure
│   ├── style.css       ← Styling & Animations
│   └── script.js       ← Client-Side Logic & SSE Handling
├── requirements.txt    ← Project Dependencies
└── .gitignore          ← Excluded files & directories
```

---

## 🚀 Getting Started

Follow these steps to run the full system on your local machine.

### 1. Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Ollama** (for local AI inference)

### 2. Start the AI Model
Open a separate terminal and start Ollama (ensure Qwen 2.5 is pulled/configured as needed):
```bash
ollama serve
```

### 3. Setup the Environment & Run the Server
Open your primary terminal and run:

```bash
# Clone the repository and navigate into the root directory
cd AI_Wealth_Advisor

# Create a virtual environment (Recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Start the FastAPI Web Server
uvicorn backend.api:app --port 8000 --reload
```

### 4. Access the Dashboard
Once the server is running, the dashboard is instantly available at:
👉 **[http://localhost:8000/](http://localhost:8000/)** (Automatically redirects to the main app dashboard)

You can also view interactive API documentation (Swagger UI) at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧭 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Check if Ollama and the AI Agent are ready and responsive. |
| `GET`  | `/api/market` | Instant live snapshot of global & Indian market benchmarks. |
| `POST` | `/api/analyze/stream` | **Real-time Agentic Analysis** featuring live thought process streaming. |

---

## 🧠 Core Features

- ⚡ **SSE Streaming:** Watch the AI's "Thinking" and reasoning process as it streams live to the dashboard.
- 🤖 **Controlled Agentic Loop:** A structured framework for the AI to reason, use tools, fetch data, and summarize insights.
- 📈 **Live Market Data:** Direct integration with `yfinance` to ingest real-time asset prices and market metrics.
- 🇮🇳 **Indian Market Focus:** Built-in tools configured specifically for analyzing Nifty 50, Sensex, and selecting relevant Indian Mutual Funds.

---

*Built with ❤️ to simplify financial planning.*
