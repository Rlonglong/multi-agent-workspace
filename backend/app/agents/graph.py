import asyncio
import json
import re
import os
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage, ToolMessage  # fallback
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal

from app.agents.state import WorkspaceState, AgentConfig
from app.agents.llm_router import get_llm
from langgraph.prebuilt import ToolNode, tools_condition
from app.agents.tools import write_code_file, read_code_file, execute_playwright_qa, reset_workspace_dir
from app.services.memory_service import memory_service

# Define the shared toolkit available to all engineering agents
engineering_tools = [write_code_file, read_code_file, execute_playwright_qa]
tool_node = ToolNode(engineering_tools)


# ==========================================
# Schema for PM Node Routing and Swarm Setup
# ==========================================

class SwarmRouter(BaseModel):
    """Route tasks to a specific specialized agent, or finish the turn to ask the user a question."""
    next_step: str = Field(
        description="The role of the agent to route to next (e.g., 'Frontend', 'Playwright QA'), or 'FINISH' to return control to the Human user."
    )
    direct_message: str = Field(
        description="The message or instruction to pass to the routed agent or the Human user."
    )

class AgentSuggestion(BaseModel):
    role: str = Field(description="The professional title of the suggested AI agent (e.g. 'CTO', 'Playwright QA', 'React Node Engineer')")
    model: str = Field(description="Suggested LLM tag for this role. Prefer the same provider family as the PM unless the user explicitly chose otherwise.")
    prompt: str = Field(description="A comprehensive, specialized System Prompt instructing this agent on their precise duties, including tools they should invoke.")

class SuggestAgents(BaseModel):
    """Suggest an exhaustive list of specialized AI Agent roles required to build the user's project to completion."""
    guideline: str = Field(description="A highly detailed, comprehensive markdown Software Requirements Specification (SRS) for the project. MUST include project overview, target audience, core features, technical stack, file structure, and explicit step-by-step implementation plan. Do not be brief.")
    agents: list[AgentSuggestion] = Field(description="List of required agents")


def normalize_content(raw_content) -> str:
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        collected: list[str] = []
        for part in raw_content:
            if isinstance(part, str):
                collected.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    collected.append(str(text))
            else:
                collected.append(str(part))
        return "".join(collected)
    if isinstance(raw_content, dict):
        return str(raw_content.get("text") or raw_content.get("content") or raw_content)
    return str(raw_content)


def extract_write_code_actions(text: str) -> list[dict]:
    actions: list[dict] = []
    if not text:
        return actions

    def collect_actions(parsed):
        if isinstance(parsed, list):
            for item in parsed:
                collect_actions(item)
            return
        if not isinstance(parsed, dict):
            return
        if parsed.get("name") == "write_code_file" and isinstance(parsed.get("arguments"), dict):
            actions.append(parsed)
            return
        if "filepath" in parsed and "content" in parsed:
            actions.append({
                "name": "write_code_file",
                "arguments": {
                    "filepath": parsed.get("filepath"),
                    "content": parsed.get("content"),
                },
            })
            return
        if isinstance(parsed.get("writes"), list):
            collect_actions(parsed.get("writes"))

    block_pattern = re.compile(
        r"<<<WRITE_FILE:(?P<filepath>[^\n>]+)>>>\s*\n(?P<content>[\s\S]*?)\n<<<END_WRITE_FILE>>>",
        re.DOTALL,
    )
    for match in block_pattern.finditer(text):
        filepath = (match.group("filepath") or "").strip()
        content = match.group("content") or ""
        if filepath:
            actions.append({
                "name": "write_code_file",
                "arguments": {
                    "filepath": filepath,
                    "content": content,
                },
            })

    fenced_file_pattern = re.compile(
        r"(?:^|\n)(?:#{1,6}\s*|[-*]\s*|檔案[:：]?\s*|文件[:：]?\s*|path[:：]?\s*|filepath[:：]?\s*)?"
        r"`?(?P<filepath>(?:frontend|backend|client|server|src|app|docs)[^\n`]*?\.[A-Za-z0-9._-]+|README\.md|package\.json|docker-compose\.yml|\.env(?:\.example)?)`?"
        r"\s*\n```[A-Za-z0-9_-]*\n(?P<content>[\s\S]*?)\n```",
        re.DOTALL,
    )
    for match in fenced_file_pattern.finditer(text):
        filepath = (match.group("filepath") or "").strip()
        content = match.group("content") or ""
        if filepath:
            actions.append({
                "name": "write_code_file",
                "arguments": {
                    "filepath": filepath,
                    "content": content,
                },
            })

    candidates: list[str] = []
    for match in re.finditer(r"```(?:json|javascript|typescript|text)?\s*([\s\S]*?)```", text, re.DOTALL):
        candidates.append(match.group(1).strip())
    if text.strip().startswith("{") or text.strip().startswith("["):
        candidates.append(text.strip())

    # Handle multiple types of markdown blocks and loose JSON
    patterns = [
        r'(\{\s*"name"\s*:\s*"write_code_file"[^\}]*?"arguments"\s*:\s*\{.*?\}.*?\})',
        r'(\[\s*\{[\s\S]*?"filepath"\s*:\s*".*?[\s\S]*?\}\s*\])',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            candidates.append(match.group(1).strip())

    for candidate in candidates:
        if not candidate:
            continue
        parsed = None
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                parsed = json.loads(attempt)
                break
            except Exception:
                parsed = None
        if parsed is None:
            continue
        collect_actions(parsed)
    
    # deduplicate by filepath if possible
    unique_actions = []
    seen_paths = set()
    for act in actions:
        args = act.get("arguments", {})
        if isinstance(args, dict):
            path = args.get("filepath")
            if path and path not in seen_paths:
                seen_paths.add(path)
                unique_actions.append(act)
    return unique_actions


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_recent_written_files(messages: list) -> list[str]:
    found: list[str] = []
    for message in messages[-8:]:
        content = normalize_content(getattr(message, "content", ""))
        for match in re.finditer(r"Successfully wrote file:\s*(.+)", content):
            path = match.group(1).strip()
            if path and path not in found:
                found.append(path)
    return found


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def build_implementation_summary(guideline: str, agent_count: int) -> str:
    return (
        "我已整理好 implementation guideline 與預設 agent team。\n\n"
        f"- 已生成可編輯 guideline\n"
        f"- 已配置 {agent_count} 位 agents 的初始角色與 prompt\n"
        "- 請在中央確認 implementation 內容，並在右側調整 agents 後再開始 execution"
    )


def is_reviewer_role(role: str) -> bool:
    lowered = (role or "").lower()
    return any(keyword in lowered for keyword in [
        # English
        "qa", "review", "tester", "test", "checker", "audit", "verify",
        # Chinese
        "測試", "驗收", "審查", "檢查", "評審",
    ])


def is_engineering_role(role: str) -> bool:
    lowered = (role or "").lower()
    if not lowered or lowered == "pm" or is_reviewer_role(role):
        return False
    return any(keyword in lowered for keyword in [
        # English
        "cto", "engineer", "developer", "frontend", "backend", "fullstack", "tech", "platform", "infra",
        # Chinese
        "工程師", "開發者", "前端", "後端", "全端", "架構師", "資深工程", "軟體", "系統",
    ])


def requires_file_delivery(role: str) -> bool:
    lowered = (role or "").lower()
    return any(
        keyword in lowered
        for keyword in [
            # English
            "frontend", "backend", "fullstack", "developer", "engineer", "devops", "platform", "infra",
            # Chinese
            "前端", "後端", "全端", "工程師", "開發者", "系統",
        ]
    ) and not any(keyword in lowered for keyword in [
        "cto", "architect", "tech lead", "lead",
        "架構師", "技術長", "資深",
    ])


def role_priority(role: str) -> int:
    lowered = (role or "").lower()
    if is_reviewer_role(role):
        return 4
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead", "架構師", "技術長"]):
        return 0
    if any(keyword in lowered for keyword in ["backend", "server", "api", "後端"]):
        return 1
    if any(keyword in lowered for keyword in ["frontend", "ui", "web", "前端"]):
        return 2
    if any(keyword in lowered for keyword in ["marketing", "growth", "seo", "sales", "行銷", "成長"]):
        return 3
    return 2


def model_provider_family(model: str) -> str:
    lowered = (model or "").lower().strip()
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith("gpt-") or lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
        return "openai"
    if lowered.startswith("ollama/"):
        return "ollama"
    if lowered.startswith("groq/"):
        return "groq"
    if lowered.startswith("openrouter/"):
        return "openrouter"
    if lowered.startswith("claude"):
        return "anthropic"
    return "unknown"


def provider_default_model(role: str, provider_family: str, pm_model_str: str) -> str:
    if provider_family == "gemini":
        if is_reviewer_role(role):
            return "gemini-2.5-flash"
        if any(keyword in (role or "").lower() for keyword in ["cto", "architect", "tech lead", "lead", "manager"]):
            return "gemini-2.5-flash"
        return "gemini-2.0-flash-001"
    if provider_family == "openai":
        if is_reviewer_role(role):
            return "gpt-4o"
        if any(keyword in (role or "").lower() for keyword in ["cto", "architect", "tech lead", "lead", "manager"]):
            return "gpt-4o"
        return "gpt-4o-mini"
    if provider_family == "ollama":
        if is_reviewer_role(role):
            return "ollama/deepseek-r1:32b"
        if any(keyword in (role or "").lower() for keyword in ["cto", "architect", "tech lead", "lead", "manager", "devops"]):
            return "ollama/deepseek-r1:32b"
        return "ollama/qwen2.5"
    return pm_model_str


def coerce_model_to_provider_family(model: str, role: str, pm_model_str: str) -> str:
    current_model = (model or "").strip()
    pm_family = model_provider_family(pm_model_str)
    current_family = model_provider_family(current_model)
    if not current_model:
        return provider_default_model(role, pm_family, pm_model_str)
    if pm_family in {"gemini", "openai", "ollama"} and current_family != pm_family:
        return provider_default_model(role, pm_family, pm_model_str)
    return current_model


def fallback_model_for_role(role: str, base_model: str = "gpt-4o") -> str:
    lowered = (role or "").lower()
    lowered_model = (base_model or "").lower()
    # If base model is local, lean towards local fallbacks
    is_local = "ollama" in lowered_model
    provider_family = model_provider_family(base_model)

    if lowered_model.startswith("gemini-2.5-flash"):
        return "gemini-2.0-flash-001"
    if lowered_model.startswith("gemini-2.5-pro"):
        return "gemini-2.5-flash"
    if lowered_model.startswith("gemini-2.0-flash"):
        return "gemini-2.5-flash"
    if lowered_model.startswith("gpt-4o"):
        return "gpt-4o-mini"

    if provider_family == "gemini":
        return provider_default_model(role, "gemini", base_model)
    if provider_family == "openai":
        return provider_default_model(role, "openai", base_model)
    if provider_family == "ollama":
        return provider_default_model(role, "ollama", base_model)

    if is_reviewer_role(role):
        return "ollama/deepseek-r1:32b" if is_local else "gpt-4o"
    # CTO / architect / tech lead → strong reasoning model
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead", "manager", "devops"]):
        return "ollama/deepseek-r1:32b" if is_local else "gpt-4o"
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return "ollama/qwen2.5" if is_local else base_model
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return "ollama/qwen2.5" if is_local else base_model
    # Catch-all: any unrecognized role that happens to be using a small local model
    # should fall back to qwen2.5 (far more capable for structured output than llama3.2)
    if is_local and "llama3.2" in base_model:
        return "ollama/qwen2.5"
    return "ollama/qwen2.5" if is_local else base_model


def is_retryable_model_error(err: Exception | None) -> bool:
    if not err:
        return False
    text = str(err).lower()
    return any(marker in text for marker in [
        "503",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "rate limit",
        "temporarily unavailable",
        "deadline exceeded",
    ])


def looks_like_problem_report(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in [
        "有問題", "錯誤", "失敗", "不通過", "未通過", "需要修正", "需修正",
        "bug", "error", "fail", "fix", "缺失", "不符合", "阻塞"
    ])


def build_execution_queue(agent_configs: list[dict]) -> list[str]:
    ordered_roles: list[tuple[int, int, str]] = []
    for index, config in enumerate(agent_configs or []):
        role = (config.get("role") or "").strip()
        if not role or role == "PM":
            continue
        ordered_roles.append((role_priority(role), index, role))
    ordered_roles.sort(key=lambda item: (item[0], item[1]))
    queue: list[str] = []
    seen = set()
    for _, _, role in ordered_roles:
        if role in seen:
            continue
        seen.add(role)
        queue.append(role)
    return queue


def format_role_queue(queue: list[str]) -> str:
    if not queue:
        return "PM"
    return "PM -> " + " -> ".join(queue)


def truncate_for_pm(text: str, limit: int = 180) -> str:
    cleaned = strip_thinking_blocks(normalize_content(text)).replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def role_execution_brief(role: str) -> dict[str, str]:
    lowered = (role or "").lower()
    if lowered == "pm":
        return {
            "duty": "作為專案總監，負責主持整個 execution 階段。你的目標是產出成品，而不僅僅是協調。你需要排定最優執行的 queue，判斷每位 agent 的回報是否達到『可交付』標準。若發現實作有缺陷或 QA 不通過，你必須果斷安排回修（rem ediation）。",
            "deliverable": "清晰的下一步指令。每次回覆必須包含：(1) 目前整體完成進度 (2) 下一位執行者 (3) 具體且可執行的 task 指令 (4) 指出目前的技術風險或未完成項目。",
            "forbidden": "不要只做無意義的傳話，不要問使用者下一步該怎麼做（execution 階段由你主導），不要接受沒有實際檔案變更的工程回報。",
        }
    if any(keyword in lowered for keyword in ["qa", "review", "tester", "audit", "verify"]):
        return {
            "duty": "作為嚴苛的質量把關者。你必須對照 Guideline 進行『逐條』檢查。不只是看檔案是否存在，還要看邏輯是否正確、是否符合技術棧要求、是否有潛在 Bug。",
            "deliverable": "(1) 通過項目清單 (2) 具體錯誤報告（需附上檔案路徑與錯誤原因）(3) 修復建議 (4) 最終結論：PASS 或 FAIL。任何 FAIL 都會觸發回修流程。",
            "forbidden": "絕對禁止為了趕進度而給出模糊的『大致正確』，禁止跳過任何一條 Guideline 的檢查，禁止自己修改程式碼（請指派給開發者）。",
        }
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return {
            "duty": "負責實現所有前端 UI、互動邏輯與狀態管理。你必須確保畫面的美觀與操作的流暢度。",
            "deliverable": "(1) 實際的檔案寫入 (2) 前端功能說明 (3) 驗證方式（例如：開啟瀏覽器查看哪個頁面）。",
            "forbidden": "不要回報『我打算做什麼』，請直接回報『我已經做了什麼並寫入檔案』，不要在沒有 read_code_file 的情況下盲目覆蓋既有檔案。",
        }
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return {
            "duty": "負責伺服器端邏輯、API 設計、資料庫整合與串流處理。你必須確保後端的穩定性、安全性與高效能。",
            "deliverable": "(1) 實際的檔案寫入 (2) API 接口定義與使用說明 (3) 驗證方式（例如：使用 curl 或測試指令查看回傳值）。",
            "forbidden": "不要產出 placeholder 程式碼，不要要求 PM 決定技術細節（你自己就是專家），不要遺漏錯誤處理（error handling）。",
        }
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead"]):
        return {
            "duty": "負責架構設計與技術決策。你需要將複雜的 Guideline 拆解成具體、有先後依賴關係的任務，並確保所有 agent 的實作風格一致。",
            "deliverable": "(1) 技術規格書細化 (2) 檔案結構定義 (3) 給各角色的具體實作邊界說明。",
            "forbidden": "不要只是重複 Guideline，不要給出無法落地的抽象概念，不要在其他 agent 正在實作時隨意大幅改動底層架構。",
        }
    if any(keyword in lowered for keyword in ["devops", "platform", "infra"]):
        return {
            "duty": "負責部署設定、環境變數、啟動方式與基礎維運檔案。",
            "deliverable": "直接產出部署或啟動相關檔案，並說明如何執行。",
            "forbidden": "不要只說理想架構，不要在沒有實作內容時回報完成。",
        }
    if any(keyword in lowered for keyword in ["marketing", "growth", "seo", "sales"]):
        return {
            "duty": "負責文案、行銷定位、SEO 與對外說明內容。",
            "deliverable": "產出實際文案、SEO meta、landing 文本或說明文件。",
            "forbidden": "不要搶做工程工作，不要只講概念。",
        }
    return {
        "duty": "依角色完成自己那一段工作，並協助 PM 推進到成品。",
        "deliverable": "提供可以直接採用的內容或檔案修改，並說明下一步由誰接手。",
        "forbidden": "不要只做口頭進度報告。",
    }


def role_task_package(role: str) -> str:
    lowered = (role or "").lower()
    if lowered == "pm":
        return "主持 execution、安排順序、判斷是否需要回修，並確保最後有可驗收成品。"
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead"]):
        return "一次完成可落地的技術規格、檔案結構、模組邊界、開發順序與交接說明；內容必須足以讓工程角色直接開寫，不要只給原則。"
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return "一次完成一個可執行的後端工作包：至少同時交付核心 API 路由、controller/service、必要設定或啟動檔、基本錯誤處理與驗證方式；不要只初始化空檔。"
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return "一次完成一個可互動的前端工作包：至少同時交付主頁面、2-4 個核心元件、狀態或 service 串接、基本樣式與互動；不要只建立 placeholder。"
    if any(keyword in lowered for keyword in ["qa", "review", "tester", "audit", "verify"]):
        return "對照 guideline 逐條驗收，列出 PASS/FAIL 與具體缺口；若不通過，要明確指出要退回哪個角色修。"
    if any(keyword in lowered for keyword in ["devops", "platform", "infra"]):
        return "一次完成可執行的環境設定、啟動腳本、部署或本地運行說明，並讓其他角色拿到後可以直接啟動。"
    if any(keyword in lowered for keyword in ["marketing", "growth", "seo", "sales"]):
        return "一次完成可直接上線的文案、SEO、對外說明頁內容。"
    return "一次完成你職責範圍內最主要的一塊可交付成果，不要只做規劃。"


def role_file_expectation(role: str) -> str:
    lowered = (role or "").lower()
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return (
            "後端角色這一輪至少應同時落地 3 到 6 個互相關聯檔案，例如："
            "`backend/src/app.js`、`backend/src/routes/chatRoutes.js`、`backend/src/controllers/chatController.js`、"
            "`backend/src/services/geminiService.js`、`backend/package.json`、`.env.example`。"
        )
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return (
            "前端角色這一輪至少應同時落地 3 到 8 個互相關聯檔案，例如："
            "`frontend/src/App.jsx`、`frontend/src/pages/ChatPage.jsx`、`frontend/src/components/ChatWindow.jsx`、"
            "`frontend/src/components/MessageInput.jsx`、`frontend/src/services/apiService.js`、`frontend/src/styles/index.css`。"
        )
    if any(keyword in lowered for keyword in ["devops", "platform", "infra"]):
        return "DevOps 角色這一輪至少應同時交付 2 到 5 個可執行檔案，例如啟動腳本、部署設定、README 或 env 範本。"
    return "若這一輪需要多個檔案，請一次完成並一起交付，不要拆成多個小回合。"


def role_min_file_count(role: str) -> int:
    lowered = (role or "").lower()
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return 3
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return 3
    if any(keyword in lowered for keyword in ["devops", "platform", "infra"]):
        return 2
    return 1


def role_file_examples(role: str) -> list[str]:
    lowered = (role or "").lower()
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return [
            "backend/package.json",
            "backend/src/app.js",
            "backend/src/routes/chatRoutes.js",
            "backend/src/controllers/chatController.js",
            "backend/src/services/geminiService.js",
            "backend/.env.example",
        ]
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return [
            "frontend/package.json",
            "frontend/src/App.jsx",
            "frontend/src/pages/ChatPage.jsx",
            "frontend/src/components/ChatWindow.jsx",
            "frontend/src/components/MessageInput.jsx",
            "frontend/src/services/apiService.js",
            "frontend/src/styles/index.css",
        ]
    if any(keyword in lowered for keyword in ["devops", "platform", "infra"]):
        return [
            "README.md",
            ".env.example",
            "docker-compose.yml",
        ]
    return []


def build_worker_task_messages(state: WorkspaceState, resolved_target_role: str) -> list:
    guideline = normalize_content(state.get("guideline", ""))
    messages = list(state.get("messages", []) or [])
    latest_user_text = ""
    latest_pm_text = ""
    for message in reversed(messages):
        if not latest_user_text and getattr(message, "type", "") == "human":
            latest_user_text = normalize_content(getattr(message, "content", ""))
        if not latest_pm_text and getattr(message, "type", "") == "ai":
            latest_pm_text = normalize_content(getattr(message, "content", ""))
        if latest_user_text and latest_pm_text:
            break

    artifacts = state.get("artifacts") or {}
    existing_files = sorted(list(artifacts.keys()))
    existing_files_text = "\n".join(f"- {path}" for path in existing_files[:20]) if existing_files else "- （目前沒有已落地檔案）"
    example_paths = role_file_examples(resolved_target_role)
    example_paths_text = "\n".join(f"- {path}" for path in example_paths) if example_paths else "- 請依 guideline 自行決定"

    compact_task = (
        f"【你的角色】{resolved_target_role}\n"
        f"【使用者最新需求】\n{latest_user_text or '（無）'}\n\n"
        f"【PM 最新指示】\n{latest_pm_text or '（無）'}\n\n"
        f"【Implementation Guideline】\n{guideline or '（目前沒有 guideline）'}\n\n"
        f"【目前已存在的 workspace 檔案】\n{existing_files_text}\n\n"
        f"【這一輪建議至少涵蓋的檔案範例】\n{example_paths_text}\n\n"
        "【硬性要求】\n"
        "- 你現在是在獨立的 agent workspace 工作，不可修改主 repo。\n"
        "- 只做你這個角色的主要工作包，但一次做完整，不要拆小段。\n"
        "- 如果需要多個檔案，這一輪一次全部完成。\n"
        "- 不要回計畫、不要回待辦、不要回空殼說明，直接交可用檔案。\n"
        "- 若要寫檔，請優先輸出可解析的 write_code_file JSON，多個檔案放同一個 JSON 陣列。\n"
    )
    return [HumanMessage(content=compact_task)]


def looks_like_guideline(text: str) -> bool:
    hints = [
        "project overview", "implementation plan", "technical stack", "file structure",
        "專案概述", "核心功能", "技術棧", "文件結構", "實施計劃", "implementation guideline"
    ]
    lowered = text.lower()
    return any(hint in lowered for hint in hints) or text.count("##") >= 2


def looks_like_finalize_preamble(text: str) -> bool:
    lowered = text.lower()
    hits = 0
    if ("需求已確認" in text) or ("資訊已足夠" in text) or ("資訊足夠" in text):
        hits += 2
    if ("實施指南" in text) or ("implementation guideline" in lowered):
        hits += 1
    if ("實作指南" in text):
        hits += 1
    if ("代理人" in text) or ("agent team" in lowered) or ("agents" in lowered):
        hits += 1
    if ("ai agents" in lowered):
        hits += 1
    if ("以下是為你" in text) or ("我將為你" in text) or ("量身打造" in text):
        hits += 1
    if ("以下是完整" in text) or ("我將為您規劃" in text) or ("我已掌握足夠資訊" in text):
        hits += 1
    if ("接下來" in text) or ("我會為你" in text):
        hits += 1
    if ("我會根據這份指南" in text) or ("我會根據需求" in text):
        hits += 1
    if ("產出實作指引" in text) or ("產出實作指南" in text) or ("產出 implementation" in lowered):
        hits += 1
    if ("建議的 agents" in lowered) or ("建議的代理" in text):
        hits += 1
    return hits >= 2 and len(text) < 1200


def build_minimal_guideline_from_request(latest_human_text: str, pm_preamble: str = "") -> str:
    request_text = (latest_human_text or "使用者希望建立一個可聊天的 ChatGPT 網頁").strip()
    preamble = strip_thinking_blocks(normalize_content(pm_preamble))
    return (
        "# Implementation Guideline\n\n"
        "## 1. 專案概述\n"
        f"- 需求主題：{request_text}\n"
        "- 目標：先產出一份可直接編輯與確認的 implementation 初稿，之後再進入 agent execution。\n\n"
        "## 2. 核心功能\n"
        "- 基本聊天介面\n"
        "- 模型切換\n"
        "- system prompt / 參數設定\n"
        "- 串流回覆\n"
        "- 基本對話歷史\n\n"
        "## 3. 技術方向\n"
        "- 前端：React / Next.js\n"
        "- 後端：Node.js API route 或 Express / FastAPI 代理\n"
        "- 模型：可切換雲端 API 與本地模型\n\n"
        "## 4. 完整檔案結構\n"
        "- frontend/src/app/page.tsx\n"
        "- frontend/src/components/\n"
        "- frontend/src/app/api/chat/route.ts\n"
        "- backend/app/main.py（若採獨立後端）\n\n"
        "## 5. Agent 分工\n"
        "- PM：主持流程、安排 queue、收斂回修\n"
        "- CTO：細化技術規格與模組邊界\n"
        "- Backend Developer：API、串流、模型整合\n"
        "- Frontend Developer：聊天 UI、設定面板、狀態管理\n"
        "- QA Engineer：逐條驗收與回修建議\n\n"
        "## 6. 驗收標準\n"
        "- 可以送出訊息並看到串流回覆\n"
        "- 可以切換模型與調整設定\n"
        "- 前後端啟動方式清楚\n"
        "- QA 能列出通過與不通過項目\n\n"
        "## 7. 備註\n"
        f"- PM 前導摘要：{preamble or 'PM 已確認需求足夠，直接進入 implementation。'}\n"
    )


def build_strict_agent_prompt(role: str, guideline: str) -> str:
    profile = role_execution_brief(role)
    role_lower = role.lower()
    execution_template = (
        "【交付報告模板】\n"
        "- 當前任務狀態：已完成 / 部分完成 / 阻塞中\n"
        "- 實際產出的檔案 (File Changes)：列出真實檔案路徑\n"
        "- 本回合完成了什麼 (Completed Scope)：說明這一輪完整完成的工作包\n"
        "- 如何驗證 (Verification Steps)：給出可操作檢查方式\n"
        "- 剩餘待辦或風險 (Roadblocks)：只有真的存在才寫\n"
        "- 建議下一位接手角色：\n"
    )
    if any(keyword in role_lower for keyword in ["qa", "review", "tester", "audit", "verify"]):
        execution_template = (
            "【驗收報告模板】\n"
            "- 檢查項狀態表 (Guideline Checklist)：\n"
            "  - [ ] 條目1：通過/不通過 (原因)\n"
            "- 發現的具體問題 (Issues Found)：\n"
            "- 應指派修復的角色 (Assigned To)：\n"
            "- 最終判定 (Overall Verdict)：PASS / FAIL\n"
        )
    
    guideline_excerpt = guideline[:4000] if guideline else "目前尚未提供 guideline，請先要求 PM 補足。"
    
    return (
        f"你是專案中的 {role}。\n\n"
        "【你的核心職責 (Your Duty)】\n"
        f"- {profile['duty']}\n\n"
        "【核心交付物 (Your Deliverables)】\n"
        f"- {profile['deliverable']}\n\n"
        "【本回合工作包 (Current Work Package)】\n"
        f"- {role_task_package(role)}\n\n"
        "【絕對禁忌 (Prohibited Actions)】\n"
        f"- {profile['forbidden']}\n\n"
        "【工作原則 (Workflow Rules)】\n"
        "- **全程使用繁體中文 (Traditional Chinese)**。\n"
        "- **拒絕口頭報告**：如果你是工程角色，你必須產出實際的檔案內容。優先直接使用 `write_code_file` 工具；若無法使用工具，請輸出 Markdown JSON 區塊，例如：```json [{\"name\":\"write_code_file\",\"arguments\":{\"filepath\":\"backend/src/app.js\",\"content\":\"// code\"}}] ```。\n"
        "- **一次完成一個完整工作包**：除非真的被阻塞，否則不要只做初始化、空檔案、待辦清單或『我接下來會做什麼』。\n"
        "- **這一輪如果要求你建立多個檔案，就一次全部完成**：不要一個檔案講一次，也不要拆成很多小回合。\n"
        "- **在獨立 agent workspace 中工作**：不要修改目前正在運行的主程式碼 repo。把 agent workspace 視為你這次專案的根目錄，在裡面建立自己的 `frontend/`、`backend/`、`docs/` 等內容。\n"
        "- **遵守 guideline 的實際路徑名稱**：若 guideline 指定使用 `frontend/` 與 `backend/`，就不要自行改成 `client/`、`server/` 或其他名字。\n"
        f"- **本輪最低交付強度**：{role_file_expectation(role)}\n"
        "- **先讀後寫 (Read Before Write)**：修改任何既有檔案前，請務必先使用 `read_code_file` 閱讀其內容，避免破壞現有邏輯。\n"
        "- **回覆必須精確且可執行**：不要說『我會嘗試...』，應說『我已完成...』。若尚未完成，請明確說明阻塞原因。\n"
        "- **QA 優先原則**：QA 角色必須極度挑剔，任何不符合 Guideline 的地方都必須標註 FAIL 並退回。\n\n"
        "【交付報告要求】\n"
        f"每次發言結束時，請務必按照以下模板回報進度：\n"
        f"{execution_template}\n"
        "【專案 Guideline 與 SRS參考】\n"
        f"{guideline_excerpt}"
    )


def ensure_complete_agent_configs(suggested: list, pm_model_str: str, pm_prompt: str, guideline: str, current_configs: list = None):
    normalized = []
    
    # Create a mapping of current roles to their configs for merging
    existing_map = {c.get("role"): c for c in (current_configs or []) if c.get("role")}
    
    for item in suggested:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or "Agent"
        
        # Merge logic: prioritize existing user selections for model/apiKey
        existing = existing_map.get(role, {})
        
        raw_model = item.get("model") or existing.get("model") or fallback_model_for_role(role, pm_model_str)
        model = coerce_model_to_provider_family(raw_model, role, pm_model_str)
        apiKey = item.get("apiKey") or existing.get("apiKey") or ""
        
        prompt = normalize_content(item.get("prompt", "")).strip()
        if len(prompt) < 80:
            # If the suggested prompt is anemic, either keep existing or build new strict one
            prompt = existing.get("prompt") or build_strict_agent_prompt(role, guideline)
            
        normalized.append({
            "role": role,
            "model": model,
            "prompt": prompt,
            "apiKey": "" if model_provider_family(model) == "ollama" else apiKey,
        })

    # Ensure PM is always present and updated
    pm_config = existing_map.get("PM", {})
    if not any(c.get("role") == "PM" for c in normalized):
        normalized.insert(0, {
            "role": "PM",
            "model": pm_model_str,
            "prompt": pm_config.get("prompt") or build_strict_agent_prompt("PM", guideline),
            "apiKey": pm_config.get("apiKey", ""),
        })
    else:
        # Update existing PM prompt if it was too short
        for c in normalized:
            if c["role"] == "PM" and len(c.get("prompt", "")) < 80:
                c["prompt"] = pm_config.get("prompt") or build_strict_agent_prompt("PM", guideline)

    if not any(is_reviewer_role(c.get("role", "")) for c in normalized):
        qa_role = "QA Engineer"
        existing_qa = next((c for r, c in existing_map.items() if is_reviewer_role(r)), {})
        qa_model = coerce_model_to_provider_family(
            existing_qa.get("model") or fallback_model_for_role(qa_role, pm_model_str),
            qa_role,
            pm_model_str,
        )
        normalized.append({
            "role": qa_role,
            "model": qa_model,
            "prompt": existing_qa.get("prompt") or build_strict_agent_prompt(qa_role, guideline),
            "apiKey": "" if model_provider_family(qa_model) == "ollama" else existing_qa.get("apiKey", ""),
        })
    return normalized


def build_default_agent_team(pm_model_str: str, guideline: str, current_configs: list = None):
    defaults = [
        {"role": "PM", "model": pm_model_str},
        {"role": "CTO", "model": fallback_model_for_role("CTO", pm_model_str)},
        {"role": "Frontend Developer", "model": fallback_model_for_role("Frontend Developer", pm_model_str)},
        {"role": "Backend Developer", "model": fallback_model_for_role("Backend Developer", pm_model_str)},
        {"role": "QA Engineer", "model": fallback_model_for_role("QA Engineer", pm_model_str)},
    ]
    return ensure_complete_agent_configs(defaults, pm_model_str, "", guideline, current_configs)


def resolve_agent_role(target_role: str, agent_configs: list[dict]) -> str | None:
    normalized_target = (target_role or "").strip().lower()
    if not normalized_target:
        return None
    exact = next((c.get("role") for c in agent_configs if (c.get("role") or "").strip().lower() == normalized_target), None)
    if exact:
        return exact
    aliases = {
        "frontend developer": ["frontend engineer", "frontend dev", "frontend"],
        "frontend engineer": ["frontend developer", "frontend dev", "frontend"],
        "backend developer": ["backend engineer", "backend dev", "backend"],
        "backend engineer": ["backend developer", "backend dev", "backend"],
        "qa engineer": ["qa reviewer", "qa", "tester", "reviewer"],
        "qa reviewer": ["qa engineer", "qa", "tester", "reviewer"],
        "cto": ["tech lead", "technical lead", "architect"],
    }
    for config in agent_configs:
        role = (config.get("role") or "").strip()
        role_lower = role.lower()
        if normalized_target in aliases and role_lower in aliases[normalized_target]:
            return role
        for alias, related in aliases.items():
            if normalized_target == alias and role_lower in related:
                return role
    return None


def apply_agent_api_key(agent_config: dict | None):
    """Route the user-supplied API key to the correct provider environment variable."""
    api_key = (agent_config or {}).get("apiKey", "")
    if not api_key:
        return
    model = (agent_config or {}).get("model", "")
    if model.startswith("groq/"):
        os.environ["GROQ_API_KEY"] = api_key
    elif model.startswith("openrouter/"):
        os.environ["OPENROUTER_API_KEY"] = api_key
    elif model.startswith("gemini"):
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
    elif model.startswith("claude"):
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif model.startswith("gpt") or model.startswith("openai"):
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        # Unknown / custom model — set all common vars as fallback
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["OPENROUTER_API_KEY"] = api_key


async def force_generate_guideline(llm, pm_prompt: str, messages) -> str:
    response = await llm.ainvoke([
        SystemMessage(content=(
            pm_prompt +
            "\n\n【強制生成 Implementation Guideline 模式】\n"
            "你現在只需要輸出一份完整的 Implementation Guideline，不得再問問題，不得輸出過場文字。\n"
            "\n"
            "這份 guideline 必須足夠詳細，讓工程師 agent 直接開始實作，不需要再問任何問題。\n"
            "請用乾淨的 Markdown 格式，必須包含以下所有章節：\n"
            "\n"
            "## 1. 專案概述（Project Overview）\n"
            "- 一句話說明這個產品是什麼、解決什麼問題\n"
            "\n"
            "## 2. 目標使用者（Target Users）\n"
            "- 主要使用情境與使用者需求\n"
            "\n"
            "## 3. 核心功能清單（Core Features）\n"
            "- 每個功能要有 feature name + 簡短描述 + 是否為 MVP 必要功能\n"
            "\n"
            "## 4. 技術棧（Tech Stack）\n"
            "- 前端：框架/工具（具體版本或 latest）\n"
            "- 後端：語言、框架（具體版本）\n"
            "- 資料庫：類型、方案\n"
            "- 部署：環境設定\n"
            "\n"
            "## 5. 完整檔案結構（File Structure）\n"
            "- 列出所有檔案路徑（含資料夾層級），例如 frontend/src/components/ChatBox.jsx\n"
            "\n"
            "## 6. API 端點規格（API Spec）\n"
            "- 每個 endpoint 要包含：Method, Path, Request body, Response format\n"
            "\n"
            "## 7. 資料結構 / Schema\n"
            "- 列出主要資料物件的欄位、型別與範例值\n"
            "\n"
            "## 8. 實作步驟（Implementation Steps）\n"
            "- 按照 agent 角色拆解，每個角色的具體交付內容（不是計畫，是可驗收的 output）\n"
            "\n"
            "## 9. 驗收標準（Acceptance Criteria）\n"
            "- 逐條列出 QA 要測試的具體場景（例如：POST /api/chat 回傳 200 且包含 message 欄位）\n"
            "\n"
            "一律使用繁體中文。要有具體技術決策，不要只說『使用適當框架』這種模糊說法。"
        ))
    ] + messages)
    return strip_thinking_blocks(normalize_content(getattr(response, "content", "")))

# ==========================================
# Node Definitions
# ==========================================


async def pm_node(state: WorkspaceState, config: RunnableConfig):
    """The central PM Supervisor. Evaluates input, suggests agents if none exist, or delegates to a worker while managing workflow stages and UI visibility."""
    messages = state.get("messages", [])
    agent_configs = state.get("agent_configs", [])
    current_stage = state.get("stage", "discovery")
    execution_started = bool(state.get("execution_started", False))
    execution_queue = list(state.get("execution_queue") or [])
    execution_cursor = int(state.get("execution_cursor") or 0)
    latest_human_text = ""
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            latest_human_text = normalize_content(getattr(message, "content", ""))
            break
    latest_ai_text = ""
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            latest_ai_text = normalize_content(getattr(message, "content", ""))
            break
    user_explicitly_wants_finalize = any(
        keyword in latest_human_text
        for keyword in [
            "需求已確認", "生成 implementation", "開始實作", "確認需求", "start implementation", "go to implementation",
            "整理成初稿", "生成實作指南", "可以整理實例", "請生成實施指南", "請生成", "生成吧", "直接生成",
            "你安排", "其餘由你", "你看著辦", "看你安排", "就用這一套", "就照這個", "這套就好"
        ]
    )
    # Only allow "go ahead" keywords if the PM just asked for permission
    if not user_explicitly_wants_finalize:
        is_generic_confirm = latest_human_text.strip() in {"好", "可以", "開始", "ok", "OK", "好的", "沒問題"}
        ai_lower = latest_ai_text.lower()
        if is_generic_confirm and any(k in ai_lower for k in ["整理", "implementation", "實作指南", "實施指南", "agent", "代理"]):
            user_explicitly_wants_finalize = True
    
    # Default PM model and prompt; allow overrides via agent configs
    pm_model_str = "gpt-4o"
    pm_prompt = "You are the central AI Project Manager. Coordinate the project, talk to the user, and delegate tasks to specialized agents."
    
    pm_config = next((c for c in agent_configs if c.get("role") == "PM"), None)
    if pm_config:
        pm_model_str = pm_config.get("model", pm_model_str)
        pm_prompt = pm_config.get("prompt", pm_prompt)
        apply_agent_api_key(pm_config)
    # Instantiate LLM
        llm = get_llm(pm_model_str, temperature=0.6)
    else:
        llm = get_llm(pm_model_str, temperature=0.6) # Default case if no PM config exists
        
    # Long-Term Memory Retrieval
    if latest_human_text and len(latest_human_text) > 5 and state.get("stage", "discovery") == "discovery":
        # Only fetch memory in discovery phase to not distract execution
        try:
            memories = memory_service.query_memory(latest_human_text, n_results=1)
            if memories:
                pm_prompt += f"\n\n【Long-Term Memory Context】\n(Here is a relevant past experience or decision from your memory):\n{memories[0]}"
        except Exception as e:
            pass
    
    # Detect if PM model is a local Ollama model (cannot do reliable tool calling)
    is_local = pm_model_str.startswith("ollama/")
    last_message = messages[-1] if messages else None
    last_message_type = getattr(last_message, "type", "")

    if state.get("stage", "discovery") == "discovery" and user_explicitly_wants_finalize:
        forced_guideline = await force_generate_guideline(llm, pm_prompt, messages)
        if not looks_like_guideline(forced_guideline):
            forced_guideline = (
                "# Implementation Guideline\n\n"
                f"## 專案概述\n{latest_human_text or '使用者希望建立一個網站，請先做出可編輯初稿。'}\n\n"
                "## 核心功能\n"
                "- 產品展示\n- 下單流程\n- 基本客服\n\n"
                "## 交付要求\n"
                "- 先完成可編輯初稿\n- 右側 agents 可修改後再開始執行\n"
            )
        fallback_agents = build_default_agent_team(pm_model_str, forced_guideline)
        
        # Store this new project guideline into Long-Term Memory
        try:
            memory_content = f"Project Guideline based on request: '{latest_human_text}'.\n\n{forced_guideline}"
            memory_service.store_memory(workspace_id="general", content=memory_content)
        except Exception as e:
            pass
            
        return {
            "messages": [AIMessage(content=build_implementation_summary(forced_guideline, len(fallback_agents)))],
            "guideline": forced_guideline,
            "agent_configs": fallback_agents,
            "next": "FINISH",
            "sender": "PM",
            "stage": "implementation",
            "sidebar_visible": False,
        }

    # Discovery stage vs Routine Routing
    if not agent_configs or len(agent_configs) <= 1: 
        system_msg = SystemMessage(content=(
            pm_prompt +
            "\n\n【最高指導原則 / CRITICAL INSTRUCTIONS】\n"
            "1. 一律使用繁體中文，回覆自然、簡短、像正常 PM 對話。\n"
            "2. 目前是 discovery 階段，先透過對話釐清需求，不要過度延伸。\n"
            "3. 如果資訊足夠或使用者已經表示可以開始，就用 SuggestAgents 產生 implementation guideline 與 agents。\n"
            "4. 產生的 guideline 必須足夠詳細，讓工程師 agents 直接開始實作，不需要再問問題。\n"
            "   要包含：具體技術棧、完整檔案結構（每個路徑）、API endpoint 規格、資料 schema、每個 agent 的具體交付物。\n"
            "   禁止使用模糊說法如『使用適當框架』『視情況而定』，要有明確技術決策。\n"
            "5. 產生 agents 時，每個 agent 要有清楚的職責、限制、輸出格式與回報 PM 規則。\n"
            "6. QA / review / code-check 角色要逐條檢查，不可模糊帶過。\n"
            f"7. 如果 stage 是 execution，就不要再問使用者，直接協調 agents 開始做事。\n"
            f"8. Latest user request: {latest_human_text[:1200]}\n"
            f"9. 如果最新訊息表示需求已確認或要直接開始，請立刻 finalize。Finalize now = {user_explicitly_wants_finalize}."
        ))
        try:
            if is_local:
                # Local models bypass tool binding entirely — plain conversation only.
                local_sys = SystemMessage(content=system_msg.content + (
                    "\n\nWhen you are ready to start the project, output this JSON block at the end of your message:\n"
                    "```json\n"
                    "{\n"
                    '  "name": "SuggestAgents",\n'
                    '  "arguments": {\n'
                    '    "guideline": "SRS content here...",\n'
                    f'    "agents": [{{"role": "Frontend", "model": "{fallback_model_for_role("Frontend Developer", pm_model_str)}", "prompt": "..."}}]\n'
                    "  }\n"
                    "}\n"
                    "```"
                ))
                response = await llm.ainvoke([local_sys] + messages)
            else:
                response = await llm.bind_tools([SuggestAgents]).ainvoke([system_msg] + messages)
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"⚠️ [Model Error]: {str(e)}")],
                "next": "FINISH",
                "sender": "PM",
                "stage": state.get("stage", "discovery"),
                "sidebar_visible": state.get("sidebar_visible", False),
            }
    else:
        # Existing agents – route work based on current conversation
        roles = build_execution_queue(agent_configs)
        if current_stage == "execution" and not execution_started and last_message_type == "human" and not latest_human_text.startswith("[SYSTEM]"):
            current_stage = "implementation"
        if current_stage == "execution" and roles:
            planned_queue = execution_queue or roles
            if ((last_message_type == "human" and latest_human_text.startswith("[SYSTEM]")) or not execution_started):
                reset_workspace_dir()
                return {
                    "messages": [AIMessage(content=(
                        "我來主持這一輪 execution。\n"
                        f"執行順序：{format_role_queue(roles)}\n"
                        "每位 agent 都必須一次完成自己角色的一個完整工作包，不要只交初始化或規劃。\n"
                        "所有內容都只會寫入獨立的 agent workspace，不會修改主程式碼 repo。\n"
                        "最後我一定會交給 QA 驗收。"
                    ))],
                    "next": roles[0],
                    "sender": "PM",
                    "stage": "execution",
                    "sidebar_visible": False,
                    "artifacts": {},
                    "execution_started": True,
                    "execution_queue": roles,
                    "execution_cursor": 1 if len(roles) > 1 else len(roles),
                    "extra": {"delivery_failures": {}},
                }
            if last_message_type != "human":
                next_role = state.get("next", "FINISH")
                if next_role == "FINISH":
                    completion_text = truncate_for_pm(latest_ai_text, 260)
                    return {
                        "messages": [AIMessage(content=(
                            "我來做這一輪收尾。\n"
                            f"目前 queue 已跑完：{format_role_queue(planned_queue)}\n"
                            f"最後回報：{completion_text or '目前沒有額外回報，但此輪已完成。'}\n"
                            "若使用者沒有新指示，這一輪就先到這裡。"
                        ))],
                        "next": "FINISH",
                        "sender": "PM",
                        "stage": "execution",
                        "sidebar_visible": False,
                        "execution_started": False,
                        "execution_queue": planned_queue,
                        "execution_cursor": len(planned_queue),
                    }
                handoff_reason = truncate_for_pm(latest_ai_text)
                if is_reviewer_role(state.get("sender", "")) and looks_like_problem_report(latest_ai_text):
                    handoff_message = (
                        f"{state.get('sender', 'QA')} 發現需要修正的項目，我已重新安排修正流程。\n"
                        f"接下來請 {next_role} 直接完成需要回修的完整工作包，之後我會再帶回 QA 複檢。"
                    )
                else:
                    handoff_message = (
                        f"收到 {state.get('sender', '上一位 agent')} 的回報。"
                        f"{' 摘要：' + handoff_reason if handoff_reason else ''}\n"
                        f"下一位請 {next_role} 接手，請直接完成你角色最主要的一塊可交付成果，不要只做初始化。"
                    )
                return {
                    "messages": [AIMessage(content=handoff_message)],
                    "next": next_role,
                    "sender": "PM",
                    "stage": "execution",
                    "sidebar_visible": False,
                    "execution_started": True,
                    "execution_queue": planned_queue,
                    "execution_cursor": execution_cursor,
                }
        if current_stage == "execution" and latest_human_text.startswith("[SYSTEM]") and roles and not execution_started:
            return {
                "messages": [AIMessage(content=(
                    "我已依照角色順序排好 execution queue，請各 agent 按順序開始工作。"
                ))],
                "next": roles[0],
                "sender": "PM",
                "stage": "execution",
                "sidebar_visible": False,
                "execution_started": True,
                "execution_queue": roles,
                "execution_cursor": 1 if len(roles) > 1 else len(roles),
            }
        try:
            if is_local:
                # To prevent DeepSeek-R1 from dropping its <think> XML payload due to system prompt pressure,
                # we maintain a lightweight system message and inject the heavy roleplay rules as the first user message.
                system_msg = SystemMessage(content="You are a highly capable reasoning AI assistant.")
                
                heavy_instructions = (
                    pm_prompt +
                    "\n\n【執行規則】\n"
                    f"- 你有一個專業團隊：{roles}。\n"
                    "- 若要分派工作，請用 ROUTE:rolename: message。\n"
                    "- 一律使用繁體中文，回覆要短、清楚、可執行。\n"
                    f"- 目前 stage = {current_stage}。如果是 execution，就直接分派工作，不要再問使用者。\n"
                    "- 若你要進入專案確認完成階段，請輸出 JSON 區塊來觸發 SuggestAgents：\n"
                    "```json\n"
                    "{\n"
                    '  "name": "SuggestAgents",\n'
                    '  "arguments": {\n'
                    '    "guideline": "YOUR COMPREHENSIVE MARKDOWN GUIDELINE HERE",\n'
                    '    "agents": [{"role": "Frontend Developer", "prompt": "..."}]\n'
                    "  }\n"
                    "}\n"
                    "```\n"
                    "**重要提示**：除非使用者要求更換模型，否則在 SuggestAgents 時「不要」指定 model 欄位，這會保留使用者目前在右側介面的自定義選擇。\n\n"
                     "如果還在對話，不要輸出這個 JSON。\n\n"
                     f"【當前進度】\n"
                     f"- 正在執行專案：{current_stage}\n"
                     f"- 角色順序：{' -> '.join(roles) if roles else '無'}\n"
                     f"- 目前完成到第 {execution_cursor - 1} 位角色。\n"
                     f"- 指派中的下一位角色：{state.get('next', '無')}\n"
                     "你可以選擇回答使用者的問題，或使用 SwarmRouter 指派上述下一位 agent 開始工作。\n\n"
                     "使用者訊息：\n\n"
                )
                
                # Prepend the heavy instructions to the latest human message safely
                modified_messages = list(messages)
                if modified_messages and modified_messages[-1].type == "human":
                     modified_messages[-1] = HumanMessage(content=heavy_instructions + modified_messages[-1].content)
                     
                response_content = ""
                async for chunk in llm.astream([system_msg] + modified_messages):
                    if chunk.content:
                        response_content += chunk.content
                response = AIMessage(content=response_content)
            else:
                system_msg = SystemMessage(content=(
                    pm_prompt +
                    "\n\n【最高指導原則 / CRITICAL INSTRUCTIONS】\n"
                    "1. 一律使用繁體中文，回覆要短、清楚、可執行。\n"
                    f"2. 你有的專業角色：{roles}。\n"
                    f"3. 目前 stage = {current_stage}。如果是 execution，就先分派工作，不要再問使用者。\n"
                    f"4. 當前角色順序：{' -> '.join(roles) if roles else '無'}\n"
                    f"5. 目前完成到第 {execution_cursor - 1} 位角色。指派中的下一位：{state.get('next', '無')}\n"
                    "6. 若要分派工作，用 SwarmRouter 或 ROUTE:rolename: message。\n"
                    "7. 不要輸出冗長教條；直接說明要做什麼、遇到什麼問題、下一步是什麼。\n"
                    "8. 指派 agent 時，除非使用者要求，否則不要隨意指定模型模型 (model 欄位)，以保留使用者的自定義模型。\n"
                    "9. 一次指派給工程師的任務可以多一點，一次再讓驗收者一起查驗，不需要頻繁呼叫其他agent。如果 PM 先前安排的 queue 沒有要更改，就不需要再叫 PM 出來主持，直接按照 queue 繼續執行。"
                ))
                # For non-local, bind_tools and astream
                response_chunk = None
                async for chunk in llm.bind_tools([SwarmRouter]).astream([system_msg] + messages, config=config):
                    if response_chunk is None:
                        response_chunk = chunk
                    else:
                        response_chunk += chunk
                response = response_chunk

        except Exception as e:
            return {
                "messages": [AIMessage(content=f"⚠️ [Model Error]: {str(e)}")],
                "next": "FINISH",
                "sender": "PM",
                "stage": state.get("stage", "discovery"),
                    "sidebar_visible": state.get("sidebar_visible", False),
                }

    # Process tool calls returned by the LLM
    next_node = "FINISH"
    new_agents = []
    tool_calls = getattr(response, "tool_calls", None)
    raw_content = getattr(response, "content", "") or ""
    normalized_raw_content = normalize_content(raw_content)
    plain_content = strip_thinking_blocks(normalized_raw_content)

    # Handle local model ROUTE: prefix delegation (text-based routing instead of tool calling)
    if is_local and not tool_calls:
        route_match = re.match(r'ROUTE:(\w+):\s*(.*)', plain_content, re.DOTALL)
        if route_match:
            next_node = route_match.group(1).strip()
            msg = route_match.group(2).strip()
            return {
                "messages": [AIMessage(content=msg)],
                "next": next_node,
                "sender": "PM",
            }

    # Fallback handling for streaming models that embed JSON in text (like deepseek-r1)
    if not tool_calls and plain_content:
        # Search for any JSON block that looks like a tool call
        # Pattern: finds things that look like {"name": "...", "arguments": {...}}
        # We try to find the most specific one first
        potential_json = None
        
        # Priority 1: Markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', plain_content, re.DOTALL)
        if json_match:
            potential_json = json_match.group(1).strip()
        else:
            # Priority 2: Searching for the raw JSON structure in the text directly
            # This looks for a sequence starting with { and having "name" and "arguments" inside
            raw_json_match = re.search(r'(\{\s*"name"\s*:\s*"(?:SuggestAgents|SwarmRouter)".*?\})', plain_content, re.DOTALL)
            if raw_json_match:
                potential_json = raw_json_match.group(1).strip()

        if potential_json:
            try:
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    tool_calls = [{"name": parsed["name"], "args": parsed["arguments"], "id": "fallback"}]
            except Exception:
                pass
    
    updates = {}
    if tool_calls:
        for tc in tool_calls:
            tc_name = tc.get("name")
            tc_args = tc.get("args") or {}
            
            if tc_name == "SuggestAgents":
                # If the LLM decided to call SuggestAgents, it already judged the info
                # is sufficient — do NOT gate this behind user_explicitly_wants_finalize.
                # Forcing the user to say a magic phrase is a UX anti-pattern.
                suggested = tc_args.get("agents", [])
                guideline_content = normalize_content(tc_args.get("guideline", "No guideline generated."))
                new_agents = ensure_complete_agent_configs(suggested, pm_model_str, pm_prompt, guideline_content, agent_configs)
                
                updates["stage"] = "implementation"
                updates["sidebar_visible"] = True
                
                return {
                    "messages": [AIMessage(content=build_implementation_summary(guideline_content, len(new_agents)))],
                    "guideline": guideline_content,
                    "agent_configs": new_agents,
                    "next": "FINISH",
                    "sender": "PM",
                    **updates,
                }
            elif tc_name == "SwarmRouter":
                args = tc.get("args")
                if not isinstance(args, dict):
                    # Fallback for weird argument formats
                    try:
                        args = json.loads(str(args)) if isinstance(args, str) else {}
                    except:
                        args = {}
                
                next_node = str(args.get("next_step", "FINISH"))
                direct_msg = str(args.get("direct_message", ""))
                
                if next_node == "execution":
                    updates["stage"] = "execution"
                    updates["sidebar_visible"] = False
                
                if next_node == "FINISH":
                    return {
                        "messages": [AIMessage(content=direct_msg)],
                        "next": "FINISH",
                        "sender": "PM",
                        **updates,
                    }
                else:
                    return {
                        "messages": [AIMessage(content=f"✅ Routing to **{next_node}** agent: {direct_msg}")],
                        "next": next_node,
                        "sender": "PM",
                        **updates,
                    }
    # If the state has no stage yet, start with discovery
    if not state.get("stage"):
        updates["stage"] = "discovery"
        updates["sidebar_visible"] = False

    if (current_stage == "discovery" and not tool_calls
            and user_explicitly_wants_finalize and not looks_like_guideline(plain_content)):
        forced_guideline = await force_generate_guideline(llm, pm_prompt, messages)
        if not looks_like_guideline(forced_guideline):
            forced_guideline = build_minimal_guideline_from_request(latest_human_text, plain_content)
        fallback_agents = build_default_agent_team(pm_model_str, forced_guideline)
        return {
            "messages": [AIMessage(content=build_implementation_summary(forced_guideline, len(fallback_agents)))],
            "guideline": forced_guideline,
            "agent_configs": fallback_agents,
            "next": "FINISH",
            "sender": "PM",
            "stage": "implementation",
            "sidebar_visible": True,
        }

    # If PM output is only a "I'm about to generate guideline + agents" preamble, convert it to a real implementation package.
    # Once the PM itself indicates the requirements are confirmed, we should finalize immediately
    # instead of forcing the user to type another magic phrase like "請生成".
    if (current_stage == "discovery" and not tool_calls and plain_content
            and looks_like_finalize_preamble(plain_content)):
        forced_guideline = await force_generate_guideline(llm, pm_prompt, messages)
        if not looks_like_guideline(forced_guideline):
            forced_guideline = build_minimal_guideline_from_request(latest_human_text, plain_content)
        fallback_agents = build_default_agent_team(pm_model_str, forced_guideline, agent_configs)
        return {
            "messages": [AIMessage(content=build_implementation_summary(forced_guideline, len(fallback_agents)))],
            "guideline": forced_guideline,
            "agent_configs": fallback_agents,
            "next": "FINISH",
            "sender": "PM",
            "stage": "implementation",
            "sidebar_visible": True,
        }

    if (current_stage == "discovery" and not tool_calls and plain_content
            and (
                (looks_like_guideline(plain_content) and len(plain_content) > 900)
                or (user_explicitly_wants_finalize and looks_like_guideline(plain_content) and len(plain_content) > 300)
            )):
        fallback_agents = build_default_agent_team(pm_model_str, plain_content)
        return {
            "messages": [AIMessage(content=build_implementation_summary(plain_content, len(fallback_agents)))],
            "guideline": plain_content,
            "agent_configs": fallback_agents,
            "next": "FINISH",
            "sender": "PM",
            "stage": "implementation",
            "sidebar_visible": True,
        }

    # If the LLM returned plain text (no tool call), surface it directly
    plain_content = normalized_raw_content
    if plain_content and not tool_calls:
        return {
            "messages": [AIMessage(content=plain_content)],
            "agent_configs": new_agents if new_agents else state.get("agent_configs"),
            "next": "FINISH",
            "sender": "PM",
            **updates,
        }

    return {
        "messages": [response],
        "agent_configs": new_agents if new_agents else state.get("agent_configs"),
        "next": next_node,
        "sender": "PM",
        **updates,
    }
    

async def worker_node(state: WorkspaceState, config: RunnableConfig):
    """Generic worker node that executes utilizing the settings for the specific agent requested."""
    # We must remember who is currently active. If this is a loop-back from a ToolNode, `next` might still be the target role.
    # The active target role is typically stored in `next` by the PM, or in `sender` if it was already processing.
    active_messages = state.get("messages", [])
    looping_from_tools = bool(active_messages and getattr(active_messages[-1], "type", "") == "tool")
    if looping_from_tools:
        target_role = state.get("sender")  # It's looping back to the sender
    else:
        target_role = state.get("next")
        
    agent_configs = state.get("agent_configs", [])
    execution_started = bool(state.get("execution_started", False))
    execution_queue = list(state.get("execution_queue") or [])
    execution_cursor = int(state.get("execution_cursor") or 0)
    resolved_target_role = resolve_agent_role(target_role, agent_configs) or target_role
    target_config = next((c for c in agent_configs if c.get("role") == resolved_target_role), None)
    
    if not target_config:
        return {
        "messages": [AIMessage(content=f"Error: Agent '{resolved_target_role}' not found in team roster.")],
            "next": "FINISH",
            "sender": resolved_target_role,
            "stage": state.get("stage"),
            "sidebar_visible": state.get("sidebar_visible", False),
            "execution_started": state.get("execution_started", False),
            "execution_queue": list(state.get("execution_queue") or []),
            "execution_cursor": int(state.get("execution_cursor") or 0),
        }
        
    apply_agent_api_key(target_config)
    primary_model = target_config.get("model", "gpt-4o")
    worker_temperature = 0.2 if is_engineering_role(resolved_target_role) else 0.5
    llm = get_llm(primary_model, temperature=worker_temperature)
    worker_llm = llm
    is_local_primary_model = primary_model.startswith("ollama/")
    if is_engineering_role(resolved_target_role) and not is_local_primary_model:
        try:
            worker_llm = llm.bind_tools(engineering_tools)
        except Exception:
            worker_llm = llm
    queue = execution_queue or [c["role"] for c in agent_configs if c.get("role") != "PM"]
    next_role = "FINISH"
    next_cursor = execution_cursor
    artifacts = dict(state.get("artifacts") or {})
    extra = dict(state.get("extra") or {})
    delivery_failures = dict(extra.get("delivery_failures") or {})
    # Only delay for Ollama (local) models — cloud APIs (Groq, Gemini, OpenAI) have
    # their own rate limiting and don't need artificial sleep between agent calls.
    if state.get("stage") == "execution" and execution_started:
        if primary_model.startswith("ollama/"):
            await asyncio.sleep(0.45)
    worker_messages = build_worker_task_messages(state, resolved_target_role) if is_engineering_role(resolved_target_role) else state["messages"]

    is_cloud_model = not primary_model.startswith("ollama/")
    system_msg = SystemMessage(content=(
        f"You are the '{resolved_target_role}' agent.\n"
        f"{target_config.get('prompt', '')}\n\n"
        "【回覆規則 / Response Rules】\n"
        "1. 使用繁體中文回覆 / Reply in Traditional Chinese.\n"
        "2. 回覆要清楚、可執行，不要空話 / Be clear and actionable.\n"
        "3. 這一輪請一次完成一個完整工作包，不要只做初始化、待辦清單、規劃文件或 placeholder。\n"
        "   In this turn, deliver ONE complete work package. NO skeleton code, NO placeholder comments, NO TODO lists.\n"
        "4. 如果這一輪需要建立多個檔案，請一次在同一輪全部完成，不要一個檔案交一次。\n"
        "   If multiple files are needed, write ALL of them in this single turn.\n"
        "5. 完成你的部分後再回報 PM，不要搶做其他角色的工作。\n"
        f"5.5. {role_file_expectation(resolved_target_role)}\n"
        + ("\n6. 【CRITICAL - FILE DELIVERY REQUIRED】\n"
           "   You MUST deliver actual working code files. Choose ONE method:\n"
           "   (A) Call the write_code_file tool directly (preferred for cloud models).\n"
           "   (B) If tool calling fails, output one or more raw file blocks in this exact format:\n"
           "       <<<WRITE_FILE:backend/src/app.js>>>\n"
           "       // FULL CODE HERE\n"
           "       <<<END_WRITE_FILE>>>\n"
           "       <<<WRITE_FILE:backend/src/routes/chatRoutes.js>>>\n"
           "       // FULL CODE HERE\n"
           "       <<<END_WRITE_FILE>>>\n"
           "   (C) Legacy fallback: output a parseable JSON block:\n"
           "       ```json\n"
           "       [\n"
           "         {\"name\":\"write_code_file\",\"arguments\":{\"filepath\":\"backend/src/app.js\",\"content\":\"// FULL CODE HERE\"}},\n"
           "         {\"name\":\"write_code_file\",\"arguments\":{\"filepath\":\"backend/src/routes/chatRoutes.js\",\"content\":\"// FULL CODE HERE\"}}\n"
           "       ]\n"
           "       ```\n"
           "   RULES: Write COMPLETE, production-ready code. No '...' ellipsis, no placeholder comments like '// TODO' or '// add logic here'.\n"
           "   Every function must be fully implemented. Every file must be runnable as-is.\n"
           "   When the task obviously needs several files, output several write actions in one response.\n"
           "   Outputting only plans, descriptions, or partial code = FAILURE.\n"
           if requires_file_delivery(resolved_target_role) else "")
        + ("\n6. 你是架構/協調角色，請直接交付可採用的決策、拆解或驗收結論，不要只說接下來要做什麼。" if is_engineering_role(resolved_target_role) and not requires_file_delivery(resolved_target_role) else "")
        + ("\n7. 若你是 QA / Reviewer，必須逐條對照 guideline 驗收，並在不通過時明確點名要退回修正的角色。" if is_reviewer_role(resolved_target_role) else "")
        + ("\n8. 你目前使用的是本地模型。請避免長篇解釋，直接輸出多個 write_code_file JSON 動作，一次交付完整檔案集合。"
           "\n   對本地模型更建議你直接輸出多個 <<<WRITE_FILE:...>>> 區塊，這比 JSON 更穩定。"
           if is_engineering_role(resolved_target_role) and is_local_primary_model else "")
    ))

    async def invoke_with_model(model_llm, model_label: str, prefer_single_shot: bool = False):
        attempts = 3 if model_label.startswith("gemini") else 1
        last_error = None
        for attempt in range(attempts):
            response_chunk = None
            stream_error = None
            if prefer_single_shot:
                try:
                    response = await model_llm.ainvoke([system_msg] + worker_messages, config=config)
                    return AIMessage(
                        content=normalize_content(getattr(response, "content", "")),
                        tool_calls=getattr(response, "tool_calls", None) or [],
                    ), None
                except Exception as err:
                    stream_error = err
            try:
                if not prefer_single_shot:
                    async for chunk in model_llm.astream([system_msg] + worker_messages, config=config):
                        if response_chunk is None:
                            response_chunk = chunk
                        else:
                            response_chunk += chunk
            except Exception as err:
                stream_error = err
            if response_chunk is None:
                try:
                    response = await model_llm.ainvoke([system_msg] + worker_messages, config=config)
                    response_chunk = AIMessage(
                        content=normalize_content(getattr(response, "content", "")),
                        tool_calls=getattr(response, "tool_calls", None) or [],
                    )
                except Exception as err:
                    stream_error = err if stream_error is None else stream_error
            if response_chunk is not None:
                return response_chunk, None
            last_error = stream_error
            if attempt < attempts - 1 and is_retryable_model_error(stream_error):
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            break
        return None, last_error

    response_chunk, response_error = await invoke_with_model(
        worker_llm,
        primary_model,
        prefer_single_shot=requires_file_delivery(resolved_target_role),
    )
    fallback_model = fallback_model_for_role(target_role, primary_model)
    if response_chunk is None and fallback_model and fallback_model != primary_model:
        await asyncio.sleep(0.25)
        fallback_llm = get_llm(fallback_model)
        if is_engineering_role(resolved_target_role):
            try:
                fallback_llm = fallback_llm.bind_tools(engineering_tools)
            except Exception:
                pass
        response_chunk, fallback_error = await invoke_with_model(
            fallback_llm,
            fallback_model,
            prefer_single_shot=requires_file_delivery(resolved_target_role),
        )
        response_error = fallback_error or response_error

    if response_chunk is None:
        retryable_failure = is_retryable_model_error(response_error)
        next_after_error = "FINISH"
        next_after_error_cursor = next_cursor
        next_stage = state.get("stage")
        sidebar_visible = state.get("sidebar_visible", False)
        execution_active = state.get("execution_started", False)
        if retryable_failure and state.get("stage") == "execution":
            next_stage = "agent_config"
            sidebar_visible = True
            execution_active = False
        elif state.get("stage") == "execution" and execution_started and queue:
            if next_cursor < len(queue):
                next_after_error = queue[next_cursor]
                next_after_error_cursor = next_cursor + 1
            else:
                next_after_error_cursor = len(queue)
        return {
            "messages": [AIMessage(content=(
                f"【{resolved_target_role} 本輪回報】\n"
                f"本輪未取得穩定的模型輸出，因此沒有新的檔案寫入。\n"
                f"模型狀態：primary={primary_model}；fallback={fallback_model}。\n"
                f"系統訊息：{response_error}\n"
                + ("補充：這看起來像供應商暫時高負載或限流，系統已先自動重試並嘗試備援模型。\n\n" if retryable_failure else "\n")
                +
                (
                    "建議：請現在直接在右側把這位 agent 暫時改成 local 模型（例如 `ollama/qwen2.5` 或 `ollama/deepseek-r1:32b`），"
                    "或改成其他雲端模型後再重新開始 execution。本輪已先停在 agent 設定階段，方便你直接調整。"
                    if retryable_failure else
                    "備註：這則回報會先交回 PM，後續可由 PM 或 QA 判斷是否需要回修或由其他角色接手。"
                )
            ))],
            "next": next_after_error,
            "sender": resolved_target_role,
            "stage": next_stage,
            "sidebar_visible": sidebar_visible,
            "execution_started": execution_active,
            "execution_queue": queue,
            "execution_cursor": next_after_error_cursor,
        }

    if target_config.get("model", "").startswith("ollama/"):
        content = normalize_content(getattr(response_chunk, "content", ""))
        response_chunk = AIMessage(content=content, tool_calls=getattr(response_chunk, "tool_calls", None) or [])

    response_text = normalize_content(getattr(response_chunk, "content", ""))
    written_files: list[str] = []
    if state.get("stage") == "execution" and is_engineering_role(resolved_target_role):
        # Only attempt repair if we are NOT looping back from a ToolNode execution.
        # When looping_from_tools=True, the file was already written by ToolNode — firing
        # another repair prompt causes Gemini/cloud models to re-write files redundantly or loop.
        if requires_file_delivery(resolved_target_role) and not getattr(response_chunk, "tool_calls", None) and not looping_from_tools:
            repair_instruction = HumanMessage(content=(
                "你上一則回覆沒有提供任何真正的寫檔結果。\n"
                "現在禁止再輸出計畫、進度、解釋、條列摘要。\n"
                "請直接做一件事：使用工具寫檔，或只輸出可解析的 write_code_file JSON。\n"
                "如果這一輪需要多個檔案，請一次全部完成。\n"
                "請優先用這種格式：\n"
                "<<<WRITE_FILE:backend/src/app.js>>>\n"
                "// code\n"
                "<<<END_WRITE_FILE>>>\n"
                "<<<WRITE_FILE:backend/src/routes/chatRoutes.js>>>\n"
                "// code\n"
                "<<<END_WRITE_FILE>>>\n"
                "如果你真的要用 JSON，也可以，但 <<<WRITE_FILE:...>>> 區塊更穩。\n"
                "檔案區塊後面可以再補 1 到 3 句你完成了什麼。"
            ))
            try:
                repaired_raw = await worker_llm.ainvoke([system_msg] + worker_messages + [repair_instruction], config=config)
                response_chunk = AIMessage(
                    content=normalize_content(getattr(repaired_raw, "content", "")),
                    tool_calls=getattr(repaired_raw, "tool_calls", None) or [],
                )
                response_text = normalize_content(getattr(response_chunk, "content", ""))
            except Exception:
                pass

        write_actions = extract_write_code_actions(response_text)
        for action in write_actions:
            args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            filepath = (args.get("filepath") or "").strip()
            content = args.get("content")
            if not filepath or content is None:
                continue
            try:
                write_code_file.invoke({
                    "filepath": filepath,
                    "content": normalize_content(content),
                })
                written_files.append(filepath)
            except Exception as write_err:
                written_files.append(f"{filepath} (write failed: {write_err})")
        if looping_from_tools:
            for path in extract_recent_written_files(state.get("messages", [])):
                if path not in written_files:
                    written_files.append(path)
        if written_files:
            for path in written_files:
                if "(write failed:" in path:
                    continue
                artifacts[path] = {"role": resolved_target_role}
            delivery_failures.pop(resolved_target_role, None)
            extra["delivery_failures"] = delivery_failures
            minimum_required_files = role_min_file_count(resolved_target_role)
            successful_written_files = [path for path in written_files if "(write failed:" not in path]
            if requires_file_delivery(resolved_target_role) and len(successful_written_files) < minimum_required_files:
                response_text = (
                    f"【{resolved_target_role} 本輪回報】\n"
                    f"本輪僅落地 {len(successful_written_files)} 個檔案，未達此角色的最低完整工作包要求（至少 {minimum_required_files} 個相關檔案）。\n"
                    f"目前已寫入：{', '.join(successful_written_files) if successful_written_files else '無'}\n\n"
                    f"{strip_thinking_blocks(response_text).strip() or '目前只有部分骨架，尚不足以視為完整交付。'}\n\n"
                    "判定：不通過，需修正。請 PM 重新安排此角色補齊同一功能所需的相關檔案後，再交由 QA 驗收。"
                ).strip()
                response_chunk = AIMessage(content=response_text)
                delivery_failures[resolved_target_role] = int(delivery_failures.get(resolved_target_role, 0)) + 1
                extra["delivery_failures"] = delivery_failures
            else:
                response_text = (
                    f"已落地實作檔案：{', '.join(written_files)}\n\n"
                    f"{strip_thinking_blocks(response_text)}"
                ).strip()
                response_chunk = AIMessage(content=response_text)
        elif requires_file_delivery(resolved_target_role):
            failure_count = int(delivery_failures.get(resolved_target_role, 0)) + 1
            delivery_failures[resolved_target_role] = failure_count
            extra["delivery_failures"] = delivery_failures
            summarized_report = strip_thinking_blocks(response_text).strip()
            if not summarized_report:
                summarized_report = "本輪沒有提供可驗證的檔案寫入內容，只留下口頭回報。"
            response_text = (
                f"【{resolved_target_role} 本輪回報】\n"
                f"{summarized_report}\n\n"
                "備註：目前系統尚未偵測到實際檔案寫入。請 PM 依整體進度決定是否繼續 queue，或交由 QA 在後段集中驗收後再退回補做。"
            )
            response_chunk = AIMessage(content=response_text)
    if state.get("stage") == "execution" and execution_started and looks_like_problem_report(response_text):
        current_role_lower = (resolved_target_role or "").lower()
        if current_role_lower == "pm" or is_reviewer_role(resolved_target_role):
            remediation_cycles = int((extra.get("remediation_cycles") or 0))
            if remediation_cycles < 2:  # cap at 2 remediation passes to prevent infinite loops
                engineering_roles = [
                    role for role in build_execution_queue(agent_configs)
                    if is_engineering_role(role) and role != resolved_target_role
                ]
                remediation_tail = engineering_roles[:]
                if any((c.get("role") or "").strip() == "PM" for c in agent_configs):
                    remediation_tail.append("PM")
                if is_reviewer_role(resolved_target_role):
                    remediation_tail.append(resolved_target_role)
                if remediation_tail:
                    queue = queue[:next_cursor] + remediation_tail + queue[next_cursor:]
                    extra["remediation_cycles"] = remediation_cycles + 1
            else:
                # Too many remediation cycles — surface the QA report and finish
                response_text = (
                    f"{response_text}\n\n"
                    "⚠️ 已達到最大修正輪數限制，本輪就此結束。請使用者手動檢查上述指出的問題。"
                )
                response_chunk = AIMessage(content=response_text)

    if state.get("stage") == "execution" and execution_started:
        if queue:
            if next_cursor < len(queue):
                next_role = queue[next_cursor]
                next_cursor = next_cursor + 1
            else:
                # Queue fully exhausted — return to PM for final wrap-up.
                # Use FINISH so pm_node enters the completion branch, not the handoff branch.
                next_role = "FINISH"
                next_cursor = len(queue)
        else:
            next_role = "PM"

    return {
        "messages": [response_chunk],
        "sender": resolved_target_role,
        "next": next_role,
        "artifacts": artifacts,
        "extra": extra,
        "stage": state.get("stage"),
        "sidebar_visible": state.get("sidebar_visible", False),
        "execution_started": state.get("execution_started", False),
        "execution_queue": queue,
        "execution_cursor": next_cursor,
    }

# ==========================================
# Graph Routing
# ==========================================

def pm_router(state: WorkspaceState) -> Literal["worker_node", "__end__"]:
    next_actor = state.get("next", "FINISH")
    if next_actor == "FINISH":
        return END
    return "worker_node"

def worker_router(state: WorkspaceState) -> Literal["tools", "pm_node", "worker_node", "__end__"]:
    """If the worker outputs tools, go to tools. Otherwise, return to PM."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    # If the LLM returned a tool call, route to tools
    if getattr(last_message, "tool_calls", None):
        return "tools"
    if state.get("stage") == "execution" and state.get("execution_started", False):
        return "pm_node"
    # Otherwise, they are done and yielded text, send control back to PM
    return "pm_node"

# Build the Graph
workflow = StateGraph(WorkspaceState)

# Add our universal nodes
workflow.add_node("pm_node", pm_node)
workflow.add_node("worker_node", worker_node)
workflow.add_node("tools", tool_node)

# The human always talks to the PM first
workflow.add_edge(START, "pm_node")

# Routing logic
workflow.add_conditional_edges(
    "pm_node",
    pm_router,
    {"worker_node": "worker_node", "__end__": END}
)

workflow.add_conditional_edges(
    "worker_node",
    worker_router,
    {"tools": "tools", "pm_node": "pm_node", "worker_node": "worker_node", "__end__": END}
)

# Tools always loop back to the worker who requested them
workflow.add_edge("tools", "worker_node")

graph = workflow.compile()
