from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.db import init_db
import uvicorn
import json
import asyncio
import traceback
import os
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage  # fallback
from app.agents.state import WorkspaceState
from app.agents.graph import graph, build_strict_agent_prompt
from app.agents.llm_router import get_llm


def normalize_content(raw_content):
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(raw_content, dict):
        return str(raw_content.get("text") or raw_content.get("content") or raw_content)
    return str(raw_content)


def format_exception_detail(err: Exception) -> str:
    detail = str(err).strip()
    if not detail:
        detail = repr(err)
    if not detail:
        detail = err.__class__.__name__
    tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
    tb_tail = "".join(tb_lines[-3:]).strip() if tb_lines else ""
    if tb_tail and tb_tail != detail:
        return f"{detail}\n{tb_tail}"
    return detail


def validate_execution_agents(agent_configs, fallback_key: str = ""):
    issues = []
    for index, agent in enumerate(agent_configs or []):
        role = agent.get("role") or f"Agent {index + 1}"
        model = (agent.get("model") or "").strip()
        key = (agent.get("apiKey") or fallback_key or "").strip()
        if not model:
            issues.append(f"{role} 缺少 model")
            continue
        if model.startswith("ollama/"):
            continue
        if not key:
            issues.append(f"{role} 缺少 API key")
            continue
        if key.startswith("AIza") and (model.startswith("gpt-") or model.startswith("claude-")):
            issues.append(f"{role} 的 key 看起來是 Google key，但 model 是 {model}")
        if model.startswith("gemini") and not key.startswith("AIza"):
            issues.append(f"{role} 的 model 是 {model}，但 API key 不是 Google key")
    return issues


def normalize_execution_agent_configs(agent_configs, workspace_model: str = "", fallback_key: str = ""):
    configs = list(agent_configs or [])
    pm_model = ""
    pm_key = ""
    for agent in configs:
        if (agent.get("role") or "").strip() == "PM":
            pm_model = (agent.get("model") or "").strip()
            pm_key = (agent.get("apiKey") or "").strip()
            break

    unified_model = pm_model or (workspace_model or "").strip() or "ollama/qwen2.5"
    unified_key = (pm_key or fallback_key or "").strip()
    normalized = []
    for agent in configs:
        role = (agent.get("role") or "").strip()
        model = unified_model
        key = unified_key
        if model.startswith("ollama/"):
            key = ""
        normalized.append({
            **agent,
            "role": role or agent.get("role") or "Agent",
            "model": model,
            "apiKey": key,
        })
    return normalized, unified_model, unified_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Multi-Agent LLM Workspace API",
    description="Backend API and WebSockets for the Multi-Agent System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "multi-agent-backend"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. Receive the initial payload from the Next.js frontend
        raw_data = await websocket.receive_text()
        payload = json.loads(raw_data)

        # 2. Inject API key into environment so LiteLLM can pick it up natively
        api_key = payload.get("api_key", "")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key

        if payload.get("type") == "generate_agent_prompt":
            agent_role = payload.get("agent_role", "Agent")
            draft_prompt = payload.get("draft_prompt", "")
            guideline = payload.get("guideline", "")
            pm_model = payload.get("model") or "gpt-4o"

            llm = get_llm(pm_model, temperature=0.4)
            generation_messages = [
                SystemMessage(
                    content=(
                        "You are a senior AI PM creating strict system prompts for specialized agents.\n"
                        "You must reply in Traditional Chinese only.\n"
                        "Return only the final system prompt text.\n"
                        "The prompt must clearly define: responsibilities, allowed actions, forbidden actions, guideline compliance, reporting format, language requirement, and escalation rules.\n"
                        "If the role is QA, reviewer, code reviewer, or tester, require line-by-line and item-by-item checking against the guideline, explicit violation reporting, and no vague approvals."
                    )
                ),
                HumanMessage(
                    content=(
                        f"角色名稱：{agent_role}\n\n"
                        f"使用者草稿：\n{draft_prompt or '（目前沒有草稿）'}\n\n"
                        f"專案 guideline：\n{guideline or '（目前沒有 guideline）'}\n\n"
                        "請輸出一份可直接作為 system prompt 的完整內容。"
                    )
                ),
            ]
            generated = await llm.ainvoke(generation_messages)
            content = normalize_content(getattr(generated, "content", ""))
            if len(content.strip()) < 120:
                content = build_strict_agent_prompt(agent_role, guideline)
            await websocket.send_text(json.dumps({
                "type": "agent_prompt_generated",
                "content": content,
                "agent_role": agent_role,
            }))
            await websocket.send_text(json.dumps({"type": "finish"}))
            return

        # 3. Convert OpenAI‑style dict messages into LangChain objects
        lc_messages = []
        for msg in payload.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        agent_configs = payload.get("agent_configs", [])
        if payload.get("stage") == "execution":
            agent_configs, unified_model, unified_key = normalize_execution_agent_configs(
                agent_configs,
                payload.get("model", ""),
                api_key,
            )
            if unified_key:
                os.environ["GEMINI_API_KEY"] = unified_key
                os.environ["GOOGLE_API_KEY"] = unified_key
                os.environ["OPENAI_API_KEY"] = unified_key
                os.environ["ANTHROPIC_API_KEY"] = unified_key

        # 4. Build the initial WorkspaceState – pydantic will fill defaults for missing fields
        state = WorkspaceState(
            messages=lc_messages,
            agent_configs=agent_configs,
            stage=payload.get("stage", "discovery"),
            sidebar_visible=payload.get("sidebar_visible", False),
            guideline=payload.get("guideline", ""),
            guideline_editable=payload.get("guideline_editable", True),
            code_blocks=payload.get("code_blocks", []),
            extra=payload.get("extra", None),
            execution_started=payload.get("execution_started", False),
            execution_queue=payload.get("execution_queue", []),
            execution_cursor=payload.get("execution_cursor", 0),
        )

        if state.get("stage") == "execution":
            issues = validate_execution_agents(agent_configs, api_key)
            if issues:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "⚠️ [Execution Blocked]: " + "；".join(issues),
                }))
                await websocket.send_text(json.dumps({"type": "finish"}))
                return

        # 5. Stream events from the LangGraph graph
        content_was_sent = False

        async for event in graph.astream_events(state, version="v2"):
            kind = event["event"]
            node_name = event.get("name", "")

            # ── Token-by-token streaming from the LLM ──
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if chunk.content:
                    await websocket.send_text(json.dumps({"type": "token", "content": chunk.content, "node": node_name}))
                    content_was_sent = True

            # ── Node completion: capture final message from any node ──
            elif kind == "on_chain_end" and node_name in ["pm_node", "worker_node"]:
                output = event["data"].get("output")
                # Debug: log every chain_end so we can diagnose issues
                print(f"[WS DEBUG] on_chain_end  name={node_name}  output_type={type(output).__name__}  has_messages={isinstance(output, dict) and 'messages' in output if output else False}")
                if output and isinstance(output, dict) and "messages" in output:
                    last_msg = output["messages"][-1]
                    content = normalize_content(getattr(last_msg, "content", None))
                    if content:
                        sender = output.get("sender", "System")
                        print(f"[WS DEBUG]   -> Sending replace: sender={sender}, content_len={len(str(content))}")
                        await websocket.send_text(json.dumps({
                            "type": "replace",
                            "content": content,
                            "node": sender,
                            "stage": output.get("stage"),
                            "sidebar_visible": output.get("sidebar_visible"),
                            "agent_configs": output.get("agent_configs"),
                            "guideline": output.get("guideline"),
                            "execution_started": output.get("execution_started"),
                            "execution_queue": output.get("execution_queue"),
                            "execution_cursor": output.get("execution_cursor"),
                        }))
                        content_was_sent = True

        # ── Fallback: if nothing was ever sent, invoke graph once more ──
        if not content_was_sent:
            print("[WS FALLBACK] No content was streamed – running graph.ainvoke as fallback")
            try:
                result = await graph.ainvoke(state)
                if result and isinstance(result, dict):
                    msgs = result.get("messages", [])
                    # Send only NEW messages (skip the ones sent by the user)
                    for msg in reversed(msgs):
                        content = normalize_content(getattr(msg, "content", ""))
                        if content and getattr(msg, "type", "") != "human":
                            await websocket.send_text(json.dumps({
                                "type": "replace",
                                "content": content,
                                "node": result.get("sender", "PM"),
                                "stage": result.get("stage"),
                                "sidebar_visible": result.get("sidebar_visible"),
                                "agent_configs": result.get("agent_configs"),
                                "guideline": result.get("guideline"),
                                "execution_started": result.get("execution_started"),
                                "execution_queue": result.get("execution_queue"),
                                "execution_cursor": result.get("execution_cursor"),
                            }))
                            content_was_sent = True
                            break
            except Exception as fallback_err:
                detail = format_exception_detail(fallback_err)
                print(f"[WS FALLBACK ERROR] {detail}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"⚠️ [Fallback Error]: {detail}",
                }))

        if not content_was_sent:
            await websocket.send_text(json.dumps({
                "type": "replace",
                "content": "⚠️ The agent processed the request but produced no visible output. Please check the backend console for debug logs.",
                "node": "System",
            }))

        await websocket.send_text(json.dumps({"type": "finish"}))

        # 6. Handle edits from the Next.js frontend
        async for raw_edit_data in websocket.iter_text():
            edit_payload = json.loads(raw_edit_data)
            edit_type = edit_payload.get("type")

            if edit_type == "edit_guideline":
                new_guideline = edit_payload.get("content", "")
                state["guideline"] = new_guideline
                state["guideline_editable"] = False
                async for event in graph.astream_events(state, version="v2"):
                    kind = event["event"]
                    node_name = event.get("name", "")
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if getattr(chunk, "tool_call_chunks", None):
                            continue
                        
                        # Handle deepseek reasoning traces injected by Ollama's OpenAI-compatible API
                        reasoning = chunk.additional_kwargs.get("reasoning_content", "") if hasattr(chunk, "additional_kwargs") else ""
                        if reasoning:
                            # The frontend expects raw <think> tags. If this is the start of reasoning, we might need to emit the tag
                            # Actually, just emitting reasoning content wrapped in <think> per chunk would be messy unless we track state.
                            # Best approach: wrap the whole reasoning stream artificially if the frontend relies on parsing <think>
                            # Wait, the easiest way is to just emit the raw reasoning text, but we need the frontend to see <think>.
                            # Since the frontend parses `<think>` and `</think>`, we can manually track reasoning state in the websocket handler!
                            await websocket.send_text(json.dumps({"type": "reasoning_token", "content": reasoning, "node": node_name}))
                            
                        if chunk.content:
                            await websocket.send_text(json.dumps({"type": "token", "content": chunk.content, "node": node_name}))
                    elif kind == "on_chain_end":
                        output = event["data"].get("output")
                        if output and isinstance(output, dict) and "messages" in output:
                            last_msg = output["messages"][-1]
                            content = normalize_content(getattr(last_msg, "content", None))
                            if content:
                                await websocket.send_text(json.dumps({
                                    "type": "replace",
                                    "content": content,
                                    "node": output.get("sender", "System"),
                                    "stage": output.get("stage"),
                                    "sidebar_visible": output.get("sidebar_visible"),
                                    "agent_configs": output.get("agent_configs"),
                                    "guideline": output.get("guideline"),
                                    "execution_started": output.get("execution_started"),
                                    "execution_queue": output.get("execution_queue"),
                                    "execution_cursor": output.get("execution_cursor"),
                                }))
                await websocket.send_text(json.dumps({"type": "finish"}))
            elif edit_type == "end_session":
                break # Exit the edit handling loop

    except WebSocketDisconnect:
        print("WebSocket disconnected by client.")
    except Exception as e:
        detail = format_exception_detail(e)
        print(f"WebSocket Graph Execution Error: {detail}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": detail}))
        except Exception:
            pass  # client may have already disconnected
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # already closed

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
