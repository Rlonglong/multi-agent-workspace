# Multi-Agent Workspace v2.0

A web-based AI chat platform that combines a normal chat experience with a PM-led multi-agent workspace for planning, execution, and review.

## Overview

This project started from a standard AI chat web app and was upgraded into **v2.0** with:

- `Chat Mode`: single-assistant conversation with streaming output
- `Workspace Mode`: a PM agent that gathers requirements, generates an implementation guideline, proposes an agent team, and coordinates execution
- `agent_workspace/`: an isolated sandbox where agents write generated project files without modifying the main system repo

## v2.0 Features

- Long-term memory with ChromaDB-backed project recall
- Multimodal input for PDFs and images
- Auto routing between cloud and local models
- Tool use through a unified agent tool layer
- QA feedback loop for rework and validation
- Execution queue UI with visible progress and manual stop control

## Core Stack

- Frontend: Next.js 14, React 18, TypeScript
- Backend: FastAPI, WebSocket streaming, LangGraph orchestration
- Local models: Ollama (`deepseek-r1:32b`, `qwen2.5`, `llama3.2`)
- Cloud models: Gemini and OpenAI-compatible routing
- Memory: ChromaDB

## Repository Structure

```text
multi-agent-workspace/
├── frontend/                 # Next.js frontend
├── backend/                  # FastAPI backend + agent orchestration
├── docs/                     # Assignment docs and architecture notes
├── agent_workspace/          # Sandbox for generated project output
├── start_ollama_x9.sh        # Local Ollama startup script
└── README.md
```

## Run Locally

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Backend

```bash
cd backend
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

### Ollama

```bash
bash start_ollama_x9.sh
```

The script stores local models under the external-drive path configured in `start_ollama_x9.sh`.

## Security Notes

- Do not commit real API keys or `.env` files
- Use the top input field or per-agent settings to provide runtime keys
- Generated files should stay inside `agent_workspace/`

## License

MIT. See [LICENSE](./LICENSE).
