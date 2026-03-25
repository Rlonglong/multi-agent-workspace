# Multi-Agent Workspace

A web-based AI chat platform with two core modes:

- `Chat Mode`: a normal single-assistant chat experience with streaming output.
- `Workspace Mode`: a PM-led multi-agent workflow for discovery, implementation planning, agent configuration, execution, and review.

This project uses a Next.js frontend and a FastAPI backend connected over HTTP and WebSocket streams.

## Highlights

- Normal chat interface with model selection and API key input
- Workspace flow with `discovery -> implementation -> execution`
- PM agent that gathers requirements before generating implementation guidelines
- Editable agent roster with per-agent model and prompt configuration
- Multi-agent execution queue with visible progress and stop control
- Separate `agent_workspace/` sandbox so generated deliverables do not modify the main app repo
- Streaming UI with collapsible thinking blocks and execution status banner

## Tech Stack

- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS
- Backend: FastAPI, WebSocket streaming, LangGraph-based agent orchestration
- Local models: Ollama
- Cloud models: Gemini / OpenAI compatible model routing

## Project Structure

```text
multi-agent-workspace/
├── frontend/                # Next.js app
├── backend/                 # FastAPI + agent orchestration
├── agent_workspace/         # Isolated sandbox for generated project output
├── docs/                    # Project docs
└── README.md
```

## Getting Started

### 1. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at [http://localhost:3000](http://localhost:3000).

### 2. Backend

```bash
cd backend
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend health check:

```bash
curl http://127.0.0.1:8000/api/health
```

### 3. Local Models with Ollama

If you want to use local models, start Ollama first and make sure your models are available.

## Security Notes

- Do not commit real API keys.
- Keep `.env` and local secret files out of version control.
- This repo only includes example env files and client-side key entry UI.
- Generated agent output is intended to stay inside `agent_workspace/`.

## Current Status

This repository is under active development. The workspace flow, queue handling, stop control, and isolated agent sandbox are implemented, but some generated-project paths and older experimental files may still be present locally during development.

## License

MIT. See [LICENSE](./LICENSE).
