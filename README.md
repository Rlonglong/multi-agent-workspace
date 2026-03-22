# Multi-Agent LLM Web Workspace

A highly interactive Web Application built with Next.js (Frontend) and FastAPI (Backend), highlighting an advanced Multi-Agent Workflow for simulating complete software development cycles.

## Features

- **Base AI Chat**: Model selection, Custom System Prompts, Param Tuning, Streaming, and Context Memory.
- **Phase 1: Product Discovery**: AI Architect/PM proposes architectures and an editable implementation guideline.
- **Phase 2: Role-based Agents**: Task distribution among Frontend, Backend, QA, and Marketing agents. 
- **Phase 3: Sandbox & QA**: Automated code execution in secure environments (Docker/venv Python) evaluated by QA agents.
- **Phase 4: UX & Anti-Deadlock**: Collapsible code UI and Human-in-the-loop intervention preventing infinite AI-loop deadlocks.

## Structure

- `/frontend`: Next.js web application.
- `/backend`: Python FastAPI server supporting Multi-Agent orchestration (e.g., using LangGraph) and Sandbox integration.

## Getting Started
*Ensure you copy `.env.example` to `.env` in both `frontend` and `backend` directories and supply your API Keys before starting.*
