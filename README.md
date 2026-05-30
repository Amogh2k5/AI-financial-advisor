# 💰 AI Wealth Advisor
### An Agentic AI Financial Planning System — Powered by Qwen 2.5

---

## 🏗️ Project Structure

```text
AI_Wealth_Advisor/
├── backend/            <-- Python Agent & API
│   ├── api.py          ← FastAPI Web Server
│   ├── agent.py        ← Controlled Agentic Loop
│   ├── tools.py        ← Financial Toolset
│   ├── main.py         ← CLI Entry Point (Optional)
│   └── requirements.txt
├── frontend/           <-- Web Dashboard
│   ├── index.html
│   ├── style.css
│   └── script.js
└── venv/               ← Python Virtual Environment
```

---

## 🚀 How to Run (Full System)

### 1. Start Ollama (Separate Terminal)
```bash
ollama serve
```

### 2. Run the FastAPI Server
```bash
# Navigate to root
cd AI_Wealth_Advisor

# Activate venv
.\venv\Scripts\activate

# Start API
uvicorn backend.api:app --port 8000 --reload
```
The dashboard will be live at: 👉 **http://localhost:8000/** (automatically redirects to `/app/`).
You can view interactive API docs at `http://localhost:8000/docs`.

---

## 🧭 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check if Ollama and Agent are ready |
| `GET` | `/api/market` | Instant live snapshot of global benchmarks |
| `POST` | `/api/analyze/stream` | **Real-time Agentic Analysis** with live step updates |

---

## 🧠 Features
- **SSE Streaming**: Shows the AI's "Thinking" process live.
- **Controlled Agent**: Uses a fixed tool-based loop (reasons → executes → summarizes).
- **Live Data**: Ingests real-time prices via `yfinance`.
- **Indian Market Focus**: Includes Nifty 50, Sensex, and specific Indian Mutual Fund suggestions.
