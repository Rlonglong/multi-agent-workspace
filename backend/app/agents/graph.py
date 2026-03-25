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
from app.agents.tools import write_code_file, read_code_file, execute_playwright_qa

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
    model: str = Field(description="Suggested LLM tag for this role. Use 'gpt-4o' for logic, 'gemini-2.5-pro' for frontend, and 'ollama/deepseek-r1:32b' for strict local validation.")
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

    patterns = [
        r'```json\s*(\{\s*"name"\s*:\s*"write_code_file".*?\})\s*```',
        r'(\{\s*"name"\s*:\s*"write_code_file".*?\})',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            candidate = match.group(1).strip()
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("name") == "write_code_file":
                actions.append(parsed)
    return actions


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
    return any(keyword in lowered for keyword in ["qa", "review", "tester", "test", "checker", "audit", "verify"])


def is_engineering_role(role: str) -> bool:
    lowered = (role or "").lower()
    if not lowered or lowered == "pm" or is_reviewer_role(role):
        return False
    return any(keyword in lowered for keyword in ["cto", "engineer", "developer", "frontend", "backend", "fullstack", "tech", "platform", "infra"])


def role_priority(role: str) -> int:
    lowered = (role or "").lower()
    if is_reviewer_role(role):
        return 4
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead"]):
        return 0
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return 1
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return 2
    if any(keyword in lowered for keyword in ["marketing", "growth", "seo", "sales"]):
        return 3
    return 2


def fallback_model_for_role(role: str) -> str:
    lowered = (role or "").lower()
    if is_reviewer_role(role):
        return "ollama/deepseek-r1:32b"
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return "ollama/qwen2.5"
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return "ollama/qwen2.5"
    if any(keyword in lowered for keyword in ["marketing", "growth", "seo", "sales"]):
        return "ollama/llama3.2"
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead", "pm"]):
        return "ollama/qwen2.5"
    return "ollama/qwen2.5"


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
            "duty": "主持整個 execution，安排順序、處理阻塞、判斷何時回到 QA 驗收，直到有可交付成果。",
            "deliverable": "每次回覆都要包含：目前完成項、下一位 agent、要他做的具體工作、是否需要 QA 或回修。",
            "forbidden": "不要把工作丟回給使用者，不要只說空泛進度，不要在 execution 中重新做 discovery。",
        }
    if any(keyword in lowered for keyword in ["qa", "review", "tester", "audit", "verify"]):
        return {
            "duty": "逐條檢查 guideline、需求與實作結果，確認是否真的完成。",
            "deliverable": "列出通過/未通過項目、缺失、影響、需要誰修、修完後要回來複檢。",
            "forbidden": "不要因為看起來差不多就通過，不要跳過任何一條 guideline，不要自己改需求。",
        }
    if any(keyword in lowered for keyword in ["frontend", "ui", "web"]):
        return {
            "duty": "負責前端頁面、互動、UI 狀態與串流體驗的實作。",
            "deliverable": "直接產出前端檔案修改，必要時讀既有檔案後再改，回報改了哪些檔案、可執行結果與剩餘缺口。",
            "forbidden": "不要只描述會怎麼做，不要回報待辦清單取代實作，不要碰後端職責。",
        }
    if any(keyword in lowered for keyword in ["backend", "server", "api"]):
        return {
            "duty": "負責 API、資料流、模型呼叫、串流處理與伺服器端邏輯。",
            "deliverable": "直接產出後端檔案修改，必要時補上錯誤處理與路由邏輯，回報具體檔案、可執行驗證方式與剩餘缺口。",
            "forbidden": "不要只交代計畫，不要要求 PM 幫你決定已明確的技術細節，不要碰前端版面職責。",
        }
    if any(keyword in lowered for keyword in ["cto", "architect", "tech lead", "lead"]):
        return {
            "duty": "拆解技術工作、校正架構與依賴順序，協助 PM 安排工程先後。",
            "deliverable": "給 PM 一個具體技術分工與先後順序，必要時產出架構相關檔案或設定。",
            "forbidden": "不要取代 PM 做總控，不要只重複 guideline，不要要求其他角色等待沒有理由的指令。",
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
    return hits >= 2 and len(text) < 1200


def build_strict_agent_prompt(role: str, guideline: str) -> str:
    profile = role_execution_brief(role)
    role_lower = role.lower()
    execution_template = (
        "【回報模板】\n"
        "- 已完成：\n"
        "- 變更檔案：\n"
        "- 可驗證結果：\n"
        "- 目前缺口：\n"
        "- 建議下一位：\n"
    )
    if any(keyword in role_lower for keyword in ["qa", "review", "tester", "audit", "verify"]):
        execution_template = (
            "【回報模板】\n"
            "- 檢查項：\n"
            "- 結果：通過 / 不通過\n"
            "- 問題與證據：\n"
            "- 需要誰修：\n"
            "- 修完後是否需回複檢：\n"
        )
    elif any(keyword in role_lower for keyword in ["backend", "server", "api", "frontend", "ui", "web", "cto", "architect", "devops", "platform", "infra"]):
        execution_template = (
            "【回報模板】\n"
            "- 已完成：\n"
            "- 修改檔案：\n"
            "- 實作內容：\n"
            "- 驗證方式：\n"
            "- 目前缺口：\n"
            "- 下一步建議：\n"
        )
    guideline_excerpt = guideline[:3000] if guideline else "目前尚未提供 guideline，請先要求 PM 補足。"
    return (
        f"你是 {role}。\n\n"
        "【你的核心職責】\n"
        f"- {profile['duty']}\n\n"
        "【你必須交付的東西】\n"
        f"- {profile['deliverable']}\n\n"
        "【你不能做的事】\n"
        f"- {profile['forbidden']}\n\n"
        "【回覆規則】\n"
        "- 一律使用繁體中文。\n"
        "- 回覆要短、清楚、可執行。\n"
        "- 若資訊不足或需求不明，先回報 PM。\n"
        "- 只做自己的角色工作，不要搶別人的工作。\n"
        "- 不要講空話，不要輸出長篇教條。\n"
        f"- {('QA 必須逐條檢查，不可用大致正確通過。' if any(keyword in role_lower for keyword in ['qa', 'review', 'tester', 'audit', 'verify']) else '工程角色必須真的修改檔案，不接受純規劃。')}\n"
        f"{execution_template}\n"
        "【實作要求】\n"
        "- 如果你是工程角色，execution 階段不是做進度報告，而是要真的產出可落地的檔案修改。\n"
        "- 若你是工程角色，優先使用 write_code_file / read_code_file 直接修改專案檔案。\n"
        "- 若工具不可用，請用 JSON 片段明確輸出每個檔案的 filepath 與 content。\n"
        "- 沒有任何檔案修改就不算完成。\n\n"
        "【專案 Guideline】\n"
        f"{guideline_excerpt}"
    )


def ensure_complete_agent_configs(suggested: list, pm_model_str: str, pm_prompt: str, guideline: str):
    normalized = []
    for item in suggested:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or "Agent"
        model = item.get("model") or "gpt-4o"
        prompt = normalize_content(item.get("prompt", "")).strip()
        if len(prompt) < 80:
            prompt = build_strict_agent_prompt(role, guideline)
        normalized.append({
            "role": role,
            "model": model,
            "prompt": prompt,
            "apiKey": item.get("apiKey", ""),
        })

    if not any(c.get("role") == "PM" for c in normalized):
        normalized.insert(0, {
            "role": "PM",
            "model": pm_model_str,
            "prompt": build_strict_agent_prompt("PM", guideline),
            "apiKey": "",
        })
    if not any(is_reviewer_role(c.get("role", "")) for c in normalized):
        normalized.append({
            "role": "QA Engineer",
            "model": "ollama/deepseek-r1:32b",
            "prompt": build_strict_agent_prompt("QA Engineer", guideline),
            "apiKey": "",
        })
    return normalized


def build_default_agent_team(pm_model_str: str, guideline: str):
    defaults = [
        {"role": "PM", "model": pm_model_str},
        {"role": "CTO", "model": "ollama/qwen2.5"},
        {"role": "Frontend Developer", "model": "ollama/qwen2.5"},
        {"role": "Backend Developer", "model": "ollama/qwen2.5"},
        {"role": "QA Engineer", "model": "ollama/deepseek-r1:32b"},
    ]
    return ensure_complete_agent_configs(defaults, pm_model_str, "", guideline)


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
    api_key = (agent_config or {}).get("apiKey", "")
    if not api_key:
        return
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = api_key


async def force_generate_guideline(llm, pm_prompt: str, messages) -> str:
    response = await llm.ainvoke([
        SystemMessage(content=(
            pm_prompt +
            "\n\n【強制最終輸出模式】\n"
            "你現在不得再問問題，也不得輸出過場文字。"
            "請直接輸出完整 implementation guideline，格式使用乾淨 Markdown。"
            "內容至少要包含：專案概述、目標使用者、核心功能、技術棧、文件結構、實作步驟、驗收標準。"
            "一律使用繁體中文。"
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
            "需求已確認", "直接整理", "直接進入", "請直接", "start implementation", "go to implementation",
            "proceed", "都可以看你", "你決定", "幫我生成", "請生成", "你處理", "開始吧", "開始", "可以開始",
            "剩下的隨便", "隨便", "你安排", "看著辦", "照你", "你看著辦",
            "好", "ok", "OK"
        ]
    )
    if not user_explicitly_wants_finalize and latest_human_text.strip() in {"好", "可以", "開始", "ok", "OK"}:
        ai_lower = latest_ai_text.lower()
        if any(k in ai_lower for k in ["implementation guideline", "實作指南", "實施指南", "agent", "代理"]):
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
            "4. 產生的 guideline 要足以讓 agents 直接開始工作，但不要冗長到像教科書。\n"
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
                    '    "agents": [{"role": "Frontend", "model": "gpt-4o", "prompt": "..."}]\n'
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
        if current_stage == "execution" and roles:
            planned_queue = execution_queue or roles
            if ((last_message_type == "human" and latest_human_text.startswith("[SYSTEM]")) or not execution_started):
                return {
                    "messages": [AIMessage(content=(
                        "我來主持這一輪 execution。\n"
                        f"執行順序：{format_role_queue(roles)}\n"
                        "我會在每一位 agent 完成後決定下一步，最後一定會交給 QA 驗收。"
                    ))],
                    "next": roles[0],
                    "sender": "PM",
                    "stage": "execution",
                    "sidebar_visible": False,
                    "execution_started": True,
                    "execution_queue": roles,
                    "execution_cursor": 1 if len(roles) > 1 else len(roles),
                }
            if last_message_type == "human":
                return {
                    "messages": [AIMessage(content=(
                        "收到你在 execution 中的新指示，我會重新安排這一輪工作。\n"
                        f"新的執行順序：{format_role_queue(roles)}\n"
                        f"第一位先由 {roles[0]} 開始。"
                    ))],
                    "next": roles[0],
                    "sender": "PM",
                    "stage": "execution",
                    "sidebar_visible": False,
                    "execution_started": True,
                    "execution_queue": roles,
                    "execution_cursor": 1 if len(roles) > 1 else len(roles),
                }
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
                    "execution_started": True,
                    "execution_queue": planned_queue,
                    "execution_cursor": len(planned_queue),
                }
            handoff_reason = truncate_for_pm(latest_ai_text)
            if is_reviewer_role(state.get("sender", "")) and looks_like_problem_report(latest_ai_text):
                handoff_message = (
                    f"{state.get('sender', 'QA')} 發現需要修正的項目，我已重新安排修正流程。\n"
                    f"接下來請 {next_role} 先處理，之後我會再帶回 QA 複檢。"
                )
            else:
                handoff_message = (
                    f"收到 {state.get('sender', '上一位 agent')} 的回報。"
                    f"{' 摘要：' + handoff_reason if handoff_reason else ''}\n"
                    f"下一位請 {next_role} 接手。"
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
                    '    "agents": [{"role": "Frontend", "model": "gpt-4o", "prompt": "..."}]\n'
                    "  }\n"
                    "}\n"
                    "```\n"
                    "如果還在對話，不要輸出這個 JSON。\n\n"
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
                    "4. 若要分派工作，用 SwarmRouter 或 ROUTE:rolename: message。\n"
                    "5. 不要輸出冗長教條；直接說明要做什麼、遇到什麼問題、下一步是什麼。"
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
                if not user_explicitly_wants_finalize and state.get("stage", "discovery") == "discovery":
                    return {
                        "messages": [AIMessage(content=(
                            "我已經整理好 implementation 草案方向，但現在還在需求確認階段。\n\n"
                            "請先確認需求是否完整；若你要我直接進入 Implementation Review，請回覆：\n"
                            "「需求已確認，請生成 implementation」"
                        ))],
                        "next": "FINISH",
                        "sender": "PM",
                        "stage": "discovery",
                        "sidebar_visible": False,
                    }
                suggested = tc_args.get("agents", [])
                guideline_content = normalize_content(tc_args.get("guideline", "No guideline generated."))
                new_agents = ensure_complete_agent_configs(suggested, pm_model_str, pm_prompt, guideline_content)
                
                updates["stage"] = "implementation"
                updates["sidebar_visible"] = False
                
                return {
                    "messages": [AIMessage(content=build_implementation_summary(guideline_content, len(new_agents)))],
                    "guideline": guideline_content,
                    "agent_configs": new_agents,
                    "next": "FINISH",
                    "sender": "PM",
                    **updates,
                }
            elif tc["name"] == "SwarmRouter":
                # Safely extract arguments (could be dict or string)
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                next_node = args.get("next_step", "FINISH")
                if next_node == "execution":
                    updates["stage"] = "execution"
                    updates["sidebar_visible"] = False
                # Show only the direct message to the user, not the boilerplate prefix
                direct_msg = args.get('direct_message', '')
                if next_node == "FINISH":
                    return {
                        "messages": [AIMessage(content=direct_msg)],
                        "next": next_node,
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

    if (not tool_calls and (not agent_configs or len(agent_configs) <= 1)
            and user_explicitly_wants_finalize and not looks_like_guideline(plain_content)):
        forced_guideline = await force_generate_guideline(llm, pm_prompt, messages)
        if looks_like_guideline(forced_guideline):
            fallback_agents = build_default_agent_team(pm_model_str, forced_guideline)
            return {
                "messages": [AIMessage(content=build_implementation_summary(forced_guideline, len(fallback_agents)))],
                "guideline": forced_guideline,
                "agent_configs": fallback_agents,
                "next": "FINISH",
                "sender": "PM",
                "stage": "implementation",
                "sidebar_visible": False,
            }

    # If PM output is only a "I'm about to generate guideline + agents" preamble, convert it to a real implementation package.
    if (not tool_calls and plain_content and (not agent_configs or len(agent_configs) <= 1)
            and looks_like_finalize_preamble(plain_content) and not looks_like_guideline(plain_content)):
        forced_guideline = await force_generate_guideline(llm, pm_prompt, messages)
        if looks_like_guideline(forced_guideline):
            fallback_agents = build_default_agent_team(pm_model_str, forced_guideline)
            return {
                "messages": [AIMessage(content=build_implementation_summary(forced_guideline, len(fallback_agents)))],
                "guideline": forced_guideline,
                "agent_configs": fallback_agents,
                "next": "FINISH",
                "sender": "PM",
                "stage": "implementation",
                "sidebar_visible": False,
            }

    if (not tool_calls and plain_content and (not agent_configs or len(agent_configs) <= 1)
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
            "sidebar_visible": False,
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
    if active_messages and getattr(active_messages[-1], "type", "") == "tool":
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
    llm = get_llm(primary_model)
    worker_llm = llm
    if is_engineering_role(resolved_target_role):
        try:
            worker_llm = llm.bind_tools(engineering_tools)
        except Exception:
            worker_llm = llm
    queue = execution_queue or [c["role"] for c in agent_configs if c.get("role") != "PM"]
    next_role = "FINISH"
    next_cursor = execution_cursor
    if state.get("stage") == "execution" and execution_started:
        await asyncio.sleep(0.45)

    system_msg = SystemMessage(content=(
        f"You are the '{resolved_target_role}' agent.\n"
        f"{target_config.get('prompt', '')}\n\n"
        "【回覆規則】\n"
        "1. 一律使用繁體中文。\n"
        "2. 回覆要短、清楚、可執行。\n"
        "3. 做完你的部分就直接回報 PM，不要延伸到其他角色。\n"
        + ("\n4. 你是工程角色，請直接產出檔案修改或可執行的 write_code_file 指令，不要只做進度報告。" if is_engineering_role(resolved_target_role) else "")
    ))

    async def invoke_with_model(model_llm, model_label: str):
        response_chunk = None
        stream_error = None
        try:
            async for chunk in model_llm.astream([system_msg] + state["messages"], config=config):
                if response_chunk is None:
                    response_chunk = chunk
                else:
                    response_chunk += chunk
        except Exception as err:
            stream_error = err
            if "No generation chunks were returned" not in str(err) and "No generations found in stream" not in str(err):
                pass
        if response_chunk is None:
            try:
                response = await model_llm.ainvoke([system_msg] + state["messages"], config=config)
                response_chunk = AIMessage(content=normalize_content(getattr(response, "content", "")))
            except Exception as err:
                return None, err if stream_error is None else stream_error
        return response_chunk, None

    response_chunk, response_error = await invoke_with_model(worker_llm, primary_model)
    fallback_model = fallback_model_for_role(target_role)
    if response_chunk is None and fallback_model and fallback_model != primary_model:
        await asyncio.sleep(0.25)
        fallback_llm = get_llm(fallback_model)
        if is_engineering_role(resolved_target_role):
            try:
                fallback_llm = fallback_llm.bind_tools(engineering_tools)
            except Exception:
                pass
        response_chunk, fallback_error = await invoke_with_model(fallback_llm, fallback_model)
        response_error = fallback_error or response_error

    if response_chunk is None:
        next_after_error = "FINISH"
        next_after_error_cursor = next_cursor
        if state.get("stage") == "execution" and execution_started and queue:
            if next_cursor < len(queue):
                next_after_error = queue[next_cursor]
                next_after_error_cursor = next_cursor + 1
            else:
                next_after_error_cursor = len(queue)
        return {
            "messages": [AIMessage(content=(
                f"Error: Agent '{resolved_target_role}' could not generate a response. "
                f"Primary model={primary_model}, fallback model={fallback_model}. "
                f"{response_error}"
            ))],
            "next": next_after_error,
            "sender": resolved_target_role,
            "stage": state.get("stage"),
            "sidebar_visible": state.get("sidebar_visible", False),
            "execution_started": state.get("execution_started", False),
            "execution_queue": queue,
            "execution_cursor": next_after_error_cursor,
        }

    if target_config.get("model", "").startswith("ollama/"):
        content = normalize_content(getattr(response_chunk, "content", ""))
        response_chunk = AIMessage(content=content)

    response_text = normalize_content(getattr(response_chunk, "content", ""))
    if state.get("stage") == "execution" and is_engineering_role(resolved_target_role):
        write_actions = extract_write_code_actions(response_text)
        written_files: list[str] = []
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
        if written_files:
            response_text = (
                f"已落地實作檔案：{', '.join(written_files)}\n\n"
                f"{strip_thinking_blocks(response_text)}"
            ).strip()
            response_chunk = AIMessage(content=response_text)
    if state.get("stage") == "execution" and execution_started and looks_like_problem_report(response_text):
        current_role_lower = (resolved_target_role or "").lower()
        if current_role_lower == "pm" or is_reviewer_role(resolved_target_role):
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

    if state.get("stage") == "execution" and execution_started:
        if queue:
            if next_cursor < len(queue):
                next_role = queue[next_cursor]
                next_cursor = next_cursor + 1
            else:
                next_role = "FINISH"
                next_cursor = len(queue)

    return {
        "messages": [response_chunk],
        "sender": resolved_target_role,
        "next": next_role,
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
