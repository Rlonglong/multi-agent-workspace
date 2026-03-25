"use client";

import { useState, useEffect, useRef, type HTMLAttributes } from "react";
import { AgentConfig } from "@/components/ModelSelector";
import MarkdownPreview from '@uiw/react-markdown-preview';
import { Send, MessageSquare, LayoutPanelLeft, Sparkles, ChevronDown, Clock, KeyRound, Bot, Rocket, Info, X, PanelLeftClose, PanelLeftOpen, RefreshCw, Edit2, Trash2, Plus, Play, Wand2, Square } from "lucide-react";

// Agent color palette: preset for common roles, random HSL for extras
const PRESET_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  PM: { bg: 'rgba(99,102,241,0.12)', border: 'rgba(99,102,241,0.5)', text: '#818cf8' },
  Frontend: { bg: 'rgba(6,182,212,0.12)', border: 'rgba(6,182,212,0.5)', text: '#22d3ee' },
  Backend: { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.5)', text: '#34d399' },
  QA: { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.5)', text: '#fbbf24' },
  Marketing: { bg: 'rgba(236,72,153,0.12)', border: 'rgba(236,72,153,0.5)', text: '#f472b6' },
  Design: { bg: 'rgba(168,85,247,0.12)', border: 'rgba(168,85,247,0.5)', text: '#a78bfa' },
  DevOps: { bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.5)', text: '#f87171' },
};
const _randomColorCache: Record<string, typeof PRESET_COLORS['PM']> = {};
function getAgentColor(role: string) {
  if (PRESET_COLORS[role]) return PRESET_COLORS[role];
  if (_randomColorCache[role]) return _randomColorCache[role];
  const hue = (role.split('').reduce((a,c) => a + c.charCodeAt(0), 0) * 137) % 360;
  const c = { bg: `hsla(${hue},60%,50%,0.12)`, border: `hsla(${hue},60%,50%,0.5)`, text: `hsl(${hue},70%,65%)` };
  _randomColorCache[role] = c;
  return c;
}

function normalizeDisplayText(text: string) {
  if (!text) return "";
  const normalized = text.replace(/\r\n/g, "\n");
  const parts = normalized.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => {
      if (part.startsWith("```")) return part;
      return part
        .replace(/^\s*---+\s*$/gm, "")
        .replace(/\n[ \t]*\n[ \t]*\n+/g, "\n\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{4,}/g, "\n\n");
    })
    .join("");
}

function isLocalModel(model?: string) {
  return (model || "").startsWith("ollama/");
}

function extractEventContent(content: unknown) {
  if (typeof content === "string") return content.replace(/\r\n/g, "\n");
  if (Array.isArray(content)) {
    return content.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const maybeText = (item as { text?: unknown; content?: unknown }).text ?? (item as { content?: unknown }).content;
        return typeof maybeText === "string" ? maybeText : "";
      }
      return "";
    }).join("");
  }
  if (content && typeof content === "object") {
    const maybeText = (content as { text?: unknown; content?: unknown }).text ?? (content as { content?: unknown }).content;
    return typeof maybeText === "string" ? maybeText.replace(/\r\n/g, "\n") : "";
  }
  return "";
}

function sealDanglingThinkBlocks(messages: ChatMessage[]) {
  let changed = false;
  const sealed = messages.map((message) => {
    if (message.role !== "assistant") return message;
    const content = message.content || "";
    const openCount = (content.match(/<think>/g) || []).length;
    const closeCount = (content.match(/<\/think>/g) || []).length;
    if (openCount > closeCount) {
      changed = true;
      return { ...message, content: `${content}\n</think>\n` };
    }
    return message;
  });
  return { messages: sealed, changed };
}

function isPreExecutionStage(stage?: string) {
  return stage === "implementation" || stage === "agent_config";
}

function getWorkspaceExecutionModel(agents: AgentConfig[], fallbackModel: string) {
  const pmModel = agents.find((agent) => agent.role === "PM")?.model?.trim() || "";
  const firstModel = agents.find((agent) => agent.model?.trim())?.model?.trim() || "";
  return pmModel || fallbackModel || firstModel || "ollama/qwen2.5";
}

function getWorkspaceExecutionIssues(agents: AgentConfig[], fallbackKey: string, fallbackModel: string) {
  const issues: string[] = [];
  const model = getWorkspaceExecutionModel(agents, fallbackModel);
  const key = (agents.find((agent) => agent.role === "PM")?.apiKey || fallbackKey || "").trim();
  if (!model) {
    issues.push("PM 缺少 model");
  }
  if (!isLocalModel(model) && !key) {
    issues.push("Execution 缺少 API key");
  }
  const keyLooksGoogle = key.startsWith("AIza");
  const modelLooksGemini = model.startsWith("gemini");
  const modelLooksOpenAI = model.startsWith("gpt-");
  const modelLooksClaude = model.startsWith("claude-");
  if (keyLooksGoogle && (modelLooksOpenAI || modelLooksClaude)) {
    issues.push(`Execution model 是 ${model}，但 API key 看起來是 Google key。請改成 Gemini 或換對應 API key。`);
  }
  if (modelLooksGemini && key && !keyLooksGoogle) {
    issues.push(`Execution model 是 ${model}，但 API key 不是 Google key。`);
  }
  return issues;
}

export interface ChatMessage {
  id: string;
  role: "system" | "user" | "assistant" | "data";
  content: string;
  name?: string;
}

const generateId = () => Date.now().toString() + "-" + Math.random().toString(36).substring(2, 9);

export type ProjectMode = "chat" | "workspace";

export interface Session {
  id: string;
  title: string;
  mode: ProjectMode;
  model: string;
  apiKey?: string;
  agentConfigs?: AgentConfig[];
  messages: ChatMessage[];
  guideline: string | null;
  updatedAt: number;
  stage?: string;
  sidebar_visible?: boolean;
  implementationReviewPending?: boolean;
  execution_started?: boolean;
  execution_queue?: string[];
  execution_cursor?: number;
}

const SESSIONS_STORAGE_KEY = "multi-agent-sessions-v2";
const API_KEY_STORAGE_KEY = "multi-agent-apikey";
const LAST_MODEL_STORAGE_KEY = "multi-agent-last-model";
const DEFAULT_PM_PROMPT = "你是中央 PM。先用自然對話釐清需求；需求夠了就產出簡潔、可執行的 implementation guideline 與 agents。回答要短、清楚、繁體中文。";

function buildDefaultWorkspaceAgents(model: string, key: string): AgentConfig[] {
  return [{
    role: "PM",
    model: model || "ollama/deepseek-r1:32b",
    prompt: DEFAULT_PM_PROMPT,
    apiKey: key || "",
  }];
}

function ensureWorkspacePmAgent(configs: AgentConfig[] | undefined, model: string, key: string): AgentConfig[] {
  const list = [...(configs || [])];
  const pmIndex = list.findIndex((c) => c.role === "PM");
  if (pmIndex === -1) {
    list.unshift({
      role: "PM",
      model: model || "ollama/deepseek-r1:32b",
      prompt: DEFAULT_PM_PROMPT,
      apiKey: key || "",
    });
    return list;
  }
  const pm = list[pmIndex];
  list[pmIndex] = {
    ...pm,
    model: pm.model || model || "ollama/deepseek-r1:32b",
    prompt: pm.prompt || DEFAULT_PM_PROMPT,
    apiKey: pm.apiKey || key || "",
  };
  return list;
}

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [agentConfigs, setAgentConfigs] = useState<AgentConfig[]>([]);
  const [expandedMsgs, setExpandedMsgs] = useState<Set<string>>(new Set());

  const [localInput, setLocalInput] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // Per-message inline editing
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState("");

  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  // Workflow stage state
  const [stage, setStage] = useState<string>('discovery');
  
  // Custom model UI states
  const [isCustomModel, setIsCustomModel] = useState(false);
  const [customModelId, setCustomModelId] = useState("");
  const [showHelp, setShowHelp] = useState(false);
  const [guidelineContent, setGuidelineContent] = useState("");
  const [guidelineRefineInput, setGuidelineRefineInput] = useState("");
  const [isRefiningGuideline, setIsRefiningGuideline] = useState(false);
  const [pendingAgentRole, setPendingAgentRole] = useState("");
  const [thinkingContent, setThinkingContent] = useState("");
  const [showWorkspaceThinking, setShowWorkspaceThinking] = useState(false);
  const workspaceThinkingStartedAtRef = useRef<number>(0);
  const [implementationReviewPending, setImplementationReviewPending] = useState(false);
  const hasAssistantSpeechStartedRef = useRef(false);
  const thinkingHideScheduledRef = useRef(false);
  const workspaceSocketRef = useRef<WebSocket | null>(null);
  const chatAbortControllerRef = useRef<AbortController | null>(null);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);
  const [panelWidth, setPanelWidth] = useState(380);
  const panelResizeStartRef = useRef<{ x: number; width: number } | null>(null);
  const [executionBannerCollapsed, setExecutionBannerCollapsed] = useState(false);

  // Phase 2: Focus lock on sidebar
  const [focusSidebar, setFocusSidebar] = useState(false);

  // Phase 3: Expanded messages tracking (already declared above)

  // Spotlight Glassmorphism
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isMouseOver, setIsMouseOver] = useState(false);
  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const initRef = useRef(false);
  useEffect(() => {
    if (initRef.current) return; // Guard against React StrictMode double-fire
    initRef.current = true;
    setMounted(true);
    const stored = localStorage.getItem(SESSIONS_STORAGE_KEY);
    const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (storedKey) setApiKey(storedKey);

    if (stored) {
      const parsed = JSON.parse(stored) as Session[];
      // Deduplicate sessions by id and forcefully migrate old configs
      const seen = new Set<string>();
      const deduped = parsed.filter(s => {
        if (seen.has(s.id)) return false;
        seen.add(s.id);

        if (s.mode !== "workspace") {
          s.agentConfigs = [];
          s.guideline = null;
          s.stage = undefined;
          s.sidebar_visible = false;
          s.implementationReviewPending = false;
        }
        
        // Auto-fix existing PM agent configs to remove '---'
        if (s.agentConfigs) {
          s.agentConfigs = s.agentConfigs.map(c => {
             if (c.role === "PM" && c.prompt && c.prompt.includes("Use --- to separate sections.")) {
                return { ...c, prompt: c.prompt.replace("Use --- to separate sections.", "").trim() };
             }
             return c;
          });
        }
        return true;
      });
      
      if (deduped.length !== parsed.length || deduped.some(s => s.agentConfigs?.some(c => c.role === "PM" && !c.prompt?.includes("---")))) {
        localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(deduped));
      }
      
      setSessions(deduped);
      if (deduped.length > 0) {
        setCurrentSessionId(deduped[0].id);
      } else {
        createNewSession("chat");
      }
    } else {
      createNewSession("chat");
    }
  }, []);

  useEffect(() => {
    if (mounted) localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  }, [apiKey, mounted]);

  const createNewSession = (mode: ProjectMode) => {
    const lastModel = localStorage.getItem(LAST_MODEL_STORAGE_KEY) || "gpt-4o";
    const defaultWorkspaceModel = mode === "workspace" ? "ollama/deepseek-r1:32b" : lastModel;
    
    const newSession: Session = {
      id: generateId(),
      title: "New " + (mode === "workspace" ? "Project" : "Chat"),
      mode,
      model: mode === "workspace" ? defaultWorkspaceModel : lastModel,
      apiKey: localStorage.getItem(API_KEY_STORAGE_KEY) || "",
      agentConfigs: mode === "workspace"
        ? buildDefaultWorkspaceAgents(defaultWorkspaceModel, localStorage.getItem(API_KEY_STORAGE_KEY) || "")
        : [],
      messages: [],
      guideline: null,
      updatedAt: Date.now(),
      stage: mode === "workspace" ? "discovery" : undefined,
      sidebar_visible: false,
      implementationReviewPending: false,
    };
    setSessions((prev) => {
      const next = [newSession, ...prev];
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
    setAgentConfigs(newSession.agentConfigs || []);
    setCurrentSessionId(newSession.id);
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const next = sessions.filter(s => s.id !== id);
    setSessions(next);
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(next));
    if (currentSessionId === id) {
      if (next.length > 0) setCurrentSessionId(next[0].id);
      else setTimeout(() => createNewSession("chat"), 0);
    }
  };

  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const executionQueue = currentSession?.execution_queue || [];
  const executionCursor = currentSession?.execution_cursor || 0;
  const activeExecutionIndex = executionQueue.length > 0
    ? Math.min(Math.max(executionCursor - 1, 0), executionQueue.length - 1)
    : -1;
  const activeExecutionRole = activeExecutionIndex >= 0 ? executionQueue[activeExecutionIndex] : "";
  const completedExecutionCount = executionQueue.length > 0
    ? Math.min(Math.max(executionCursor - 1, 0), executionQueue.length)
    : 0;

  // Intentional: this syncs session-local UI state only when the active session changes.
  useEffect(() => {
    if (currentSession) {
      if (currentSession.apiKey !== undefined) setApiKey(currentSession.apiKey);
      let effectiveConfigs = currentSession.agentConfigs || [];
      if (currentSession.mode === "workspace") {
        effectiveConfigs = ensureWorkspacePmAgent(
          effectiveConfigs,
          currentSession.model || "ollama/deepseek-r1:32b",
          currentSession.apiKey || ""
        );
        const hadPm = (currentSession.agentConfigs || []).some((agent) => agent.role === "PM");
        if (!hadPm) {
          updateSession(currentSessionId, { agentConfigs: effectiveConfigs });
        }
      } else {
        effectiveConfigs = [];
        setIsPanelOpen(false);
        setFocusSidebar(false);
      }
      setAgentConfigs(effectiveConfigs);
      setStage(currentSession.stage || "discovery");
      setGuidelineContent(currentSession.guideline || "");
      setImplementationReviewPending(
        currentSession.implementationReviewPending ?? (currentSession.stage === "implementation" && !!currentSession.guideline)
      );
      if (currentSession.mode === "workspace") {
        setIsPanelOpen(currentSession.stage === "agent_config" || currentSession.stage === "execution");
      }

      const pmModelForWorkspace = currentSession.mode === "workspace"
        ? (effectiveConfigs.find((agent) => agent.role === "PM")?.model || currentSession.model)
        : currentSession.model;
      const isM = ["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-2.5-flash", "gemini-2.5-pro", "ollama/deepseek-r1:32b", "ollama/llama3.2", "ollama/qwen2.5"].includes(pmModelForWorkspace || "gpt-4o");
      setIsCustomModel(!isM);
      if (!isM) setCustomModelId(pmModelForWorkspace || "");
    }
  }, [currentSessionId]);

  const updateSession = (id: string, updates: Partial<Session>) => {
    setSessions((prev) => {
      const next = prev.map((s) => {
        if (s.id === id) {
          const updatedSession = { ...s, ...updates };
          if (updates.messages && updates.messages.length === 2 && s.title.startsWith("New")) {
            updatedSession.title = updates.messages[0].content.slice(0, 20) + "...";
          }
          updatedSession.updatedAt = Date.now();
          return updatedSession;
        }
        return s;
      });
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const scheduleHideWorkspaceThinking = () => {
    if (thinkingHideScheduledRef.current) return;
    thinkingHideScheduledRef.current = true;
    const elapsed = Date.now() - workspaceThinkingStartedAtRef.current;
    window.setTimeout(() => {
      setShowWorkspaceThinking(false);
      thinkingHideScheduledRef.current = false;
    }, Math.max(0, 1800 - elapsed));
  };

  const scrollMessagesToBottom = (behavior: ScrollBehavior = "auto") => {
    const container = messagesScrollRef.current;
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior });
  };

  const handleMessagesScroll = () => {
    const container = messagesScrollRef.current;
    if (!container) return;
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    isNearBottomRef.current = distanceFromBottom < 96;
  };

  const requestExecutionStop = async () => {
    if (!currentSessionId) return;
    try {
      await fetch("http://127.0.0.1:8000/api/execution/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId }),
      });
    } catch (error) {
      console.error("Failed to request execution stop", error);
    }
  };

  const stopCurrentResponse = () => {
    if (currentSession?.mode === "workspace" && workspaceSocketRef.current) {
      void requestExecutionStop();
    }
    chatAbortControllerRef.current?.abort();
    chatAbortControllerRef.current = null;
    if (workspaceSocketRef.current && workspaceSocketRef.current.readyState === WebSocket.OPEN) {
      workspaceSocketRef.current.close(1000, "stopped by user");
    }
    workspaceSocketRef.current = null;
    setIsLoading(false);
    setShowWorkspaceThinking(false);
    thinkingHideScheduledRef.current = false;
  };

  const beginPanelResize = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    panelResizeStartRef.current = {
      x: event.clientX,
      width: panelWidth,
    };
    const onMove = (moveEvent: PointerEvent) => {
      if (!panelResizeStartRef.current) return;
      const delta = panelResizeStartRef.current.x - moveEvent.clientX;
      const nextWidth = Math.min(560, Math.max(300, panelResizeStartRef.current.width + delta));
      setPanelWidth(nextWidth);
    };
    const onUp = () => {
      panelResizeStartRef.current = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const streamTextTask = async (body: Record<string, unknown>, signal?: AbortSignal) => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.body) throw new Error("No response body from server.");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    return normalizeDisplayText(text);
  };

  const generateAgentPrompt = async (config: AgentConfig, index?: number) => {
    if (!currentSession) return;
    const generated = await streamTextTask({
      model: currentSession.model,
      apiKey: config.apiKey || apiKey,
      task: "generate_agent_prompt",
      payload: {
        role: config.role,
        draftPrompt: config.prompt || "",
        guideline: guidelineContent || currentSession.guideline || "",
      },
    });
    if (index === undefined) {
      handleConfigChange({ ...config, prompt: generated });
      return;
    }
    setAgentConfigs(prev => {
      const next = prev.map((item, itemIndex) => itemIndex === index ? { ...item, prompt: generated } : item);
      updateSession(currentSessionId, { agentConfigs: next });
      return next;
    });
  };

  const refineGuidelineWithAI = async () => {
    if (!guidelineRefineInput.trim() || !currentSession) return;
    setIsRefiningGuideline(true);
    try {
      const refined = await streamTextTask({
        model: currentSession.model,
        apiKey,
        task: "refine_guideline",
        payload: {
          guideline: guidelineContent || "",
          instruction: guidelineRefineInput,
        },
      });
      setGuidelineContent(refined);
      updateSession(currentSessionId, { guideline: refined, stage: "implementation" });
      setGuidelineRefineInput("");
    } finally {
      setIsRefiningGuideline(false);
    }
  };

  const submitChat = async (e?: React.FormEvent, overrideInput?: string, overrideAgentConfigs?: AgentConfig[], overrideStage?: string) => {
    e?.preventDefault();
    const inputText = overrideInput !== undefined ? overrideInput : localInput;
    if (!inputText.trim() || isLoading || !currentSessionId || !currentSession) return;

    const isSystemCommand = inputText.startsWith("[SYSTEM]");
    const sessionStage = overrideStage || stage || currentSession.stage || "discovery";
    const sessionGuideline = guidelineContent || currentSession.guideline || "";
    let outgoingAgentConfigs = overrideAgentConfigs || agentConfigs;

    // Guard: workspace must always send at least PM config, otherwise backend falls back to gpt-4o.
    if (currentSession.mode === "workspace" && (!outgoingAgentConfigs || outgoingAgentConfigs.length === 0)) {
          outgoingAgentConfigs = currentSession.agentConfigs && currentSession.agentConfigs.length > 0
        ? currentSession.agentConfigs
        : [{
            role: "PM",
            model: currentSession.model || "ollama/deepseek-r1:32b",
            prompt: DEFAULT_PM_PROMPT,
            apiKey: apiKey || currentSession.apiKey || "",
          }];
      setAgentConfigs(outgoingAgentConfigs);
      updateSession(currentSessionId, { agentConfigs: outgoingAgentConfigs });
    }

    if (currentSession.mode === "workspace") {
      // Ensure PM uses the selected model from header/session
      const selectedModel = currentSession.model || "gpt-4o";
      const hasPm = outgoingAgentConfigs.some((agent) => agent.role === "PM");
      if (!hasPm) {
        outgoingAgentConfigs = [{
          role: "PM",
          model: selectedModel,
          prompt: DEFAULT_PM_PROMPT,
          apiKey: apiKey || currentSession.apiKey || "",
        }, ...outgoingAgentConfigs];
      } else {
        outgoingAgentConfigs = outgoingAgentConfigs.map((agent) =>
          agent.role === "PM" ? { ...agent, model: agent.model || selectedModel } : agent
        );
      }
      setAgentConfigs(outgoingAgentConfigs);
      updateSession(currentSessionId, { agentConfigs: outgoingAgentConfigs });
    }

    if (currentSession.mode === "workspace" && !isSystemCommand && isPreExecutionStage(sessionStage)) {
      return;
    }

    const sealedState = sealDanglingThinkBlocks(currentSession?.messages || []);
    const baseMessages = sealedState.messages;
    if (sealedState.changed) {
      updateSession(currentSessionId, { messages: baseMessages });
    }
    const userMsg: ChatMessage = { id: generateId(), role: "user", content: inputText };
    let newMessages = [...baseMessages, userMsg];

    updateSession(currentSessionId, { messages: newMessages });
    setLocalInput("");
    setIsLoading(true);
    hasAssistantSpeechStartedRef.current = false;
    const initialThinkingText = "正在分析需求與規劃下一步...";
    setThinkingContent(initialThinkingText);
    setShowWorkspaceThinking(currentSession.mode === "workspace");
    workspaceThinkingStartedAtRef.current = Date.now();
    thinkingHideScheduledRef.current = false;

    let assistantId = generateId();
    const initialAssistantContent = currentSession.mode === "workspace"
      ? `<think>\n${initialThinkingText}\n`
      : "";
    updateSession(currentSessionId, {
      messages: [...newMessages, { id: assistantId, role: "assistant", content: initialAssistantContent }]
    });

    try {
      if (currentSession.mode === "workspace") {
        // Connect to FastAPI WebSocket for LangGraph Swarm Agent Streaming
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//127.0.0.1:8000/ws`;
        const ws = new WebSocket(wsUrl);
        workspaceSocketRef.current = ws;

        let currentStreamText = "";
        let currentNode = "PM";
        let usedReasoningToken = false;

        ws.onopen = () => {
          ws.send(JSON.stringify({
            session_id: currentSessionId,
            messages: newMessages,
            agent_configs: outgoingAgentConfigs,
            mode: "workspace",
            api_key: apiKey,
            stage: sessionStage,
            guideline: sessionGuideline,
            execution_started: currentSession.execution_started ?? false,
            execution_queue: currentSession.execution_queue ?? [],
            execution_cursor: currentSession.execution_cursor ?? 0
          }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "reasoning_token") {
              const reasoningText = extractEventContent(data.content);
              usedReasoningToken = true;
              setThinkingContent(prev => normalizeDisplayText(prev + (prev ? "\n" : "") + reasoningText));
              if (!currentStreamText.includes("<think>")) {
                currentStreamText += "<think>\n";
              }
              currentStreamText += reasoningText;
              currentNode = data.node || currentNode;
              updateSession(currentSessionId, {
                messages: [...newMessages, { id: assistantId, role: "assistant", content: currentStreamText, name: currentNode }]
              });
            } else if (data.type === "token") {
              const tokenText = extractEventContent(data.content);
              const pmConfig = outgoingAgentConfigs.find(c => c.role === "PM");
              if (currentStreamText === "" && pmConfig?.model?.includes("deepseek-r1")) {
                currentStreamText = "<think>\n\n";
                currentStreamText += tokenText.replace("<think>", "").replace(/^\n/, "");
              } else {
                currentStreamText += tokenText;
              }
              if (usedReasoningToken && currentStreamText.includes("<think>") && !currentStreamText.includes("</think>")) {
                currentStreamText += "\n</think>\n\n";
                usedReasoningToken = false;
              }
              if (tokenText.trim() && !hasAssistantSpeechStartedRef.current) {
                hasAssistantSpeechStartedRef.current = true;
                scheduleHideWorkspaceThinking();
              }
              if (data.node && !data.node.toLowerCase().includes("ollama") && !data.node.toLowerCase().includes("chat")) {
                currentNode = data.node;
              }
              let display = normalizeDisplayText(currentStreamText);
              display = display.replace(/^R(O(U(T(E(:(\w+(:\s*)?)?)?)?)?)?)?/i, "⚙️ Routing...");
              display = display.replace(/^ROUTE:\w+:\s*/ig, "");
              updateSession(currentSessionId, {
                messages: [...newMessages, { id: assistantId, role: "assistant", content: display, name: currentNode }]
              });
            } else if (data.type === "replace") {
              currentStreamText = normalizeDisplayText(extractEventContent(data.content));
              currentNode = data.node || currentNode;
              const sealedMsg = { id: assistantId, role: "assistant" as const, content: currentStreamText, name: currentNode };
              const nextGuideline = data.guideline !== undefined ? extractEventContent(data.guideline) : (currentSession.guideline || "");
              const hasGuidelinePayload = !!nextGuideline.trim();
              const hasAgentConfigsPayload = Array.isArray(data.agent_configs) && data.agent_configs.length > 1;
              const inferredImplementation = currentSession.mode === "workspace"
                && !data.stage
                && (hasGuidelinePayload || hasAgentConfigsPayload)
                && (currentSession.stage === "discovery" || !currentSession.stage);
              const nextStage = data.stage !== undefined
                ? data.stage
                : (inferredImplementation ? "implementation" : currentSession.stage);
              const stabilizedStage = currentSession.mode === "workspace" && currentSession.stage === "execution"
                ? "execution"
                : nextStage;
              const shouldOpenImplementationReview = stabilizedStage === "implementation" && !!nextGuideline.trim();
              const nextReviewPending = shouldOpenImplementationReview
                ? true
                : (stabilizedStage === "execution" ? false : (currentSession.implementationReviewPending ?? implementationReviewPending));
              
              updateSession(currentSessionId, {
                messages: [...newMessages, sealedMsg],
                stage: stabilizedStage,
                agentConfigs: data.agent_configs !== undefined ? data.agent_configs : currentSession.agentConfigs,
                guideline: data.guideline !== undefined ? nextGuideline : currentSession.guideline,
                implementationReviewPending: nextReviewPending,
                execution_started: data.execution_started !== undefined ? data.execution_started : currentSession.execution_started,
                execution_queue: data.execution_queue !== undefined ? data.execution_queue : currentSession.execution_queue,
                execution_cursor: data.execution_cursor !== undefined ? data.execution_cursor : currentSession.execution_cursor,
              });

              if (data.agent_configs) setAgentConfigs(data.agent_configs);
              if (data.sidebar_visible !== undefined) setIsPanelOpen(!!data.sidebar_visible);
              if (data.guideline !== undefined) setGuidelineContent(nextGuideline || "");

              if (data.stage !== undefined || inferredImplementation) {
                const uiStage = currentSession.mode === "workspace" && currentSession.stage === "execution"
                  ? "execution"
                  : (data.stage !== undefined ? data.stage : "implementation");
                setStage(uiStage);
                if (uiStage === 'discovery') {
                  setIsPanelOpen(false);
                  setImplementationReviewPending(false);
                }
                if (uiStage === 'implementation') {
                  setIsPanelOpen(false);
                  setFocusSidebar(false);
                  if (shouldOpenImplementationReview) setImplementationReviewPending(true);
                }
                if (uiStage === 'agent_config') {
                  setIsPanelOpen(true);
                  setFocusSidebar(true);
                  setImplementationReviewPending(false);
                }
                if (uiStage === 'execution') {
                  setIsPanelOpen(true);
                  setFocusSidebar(false);
                  setImplementationReviewPending(false);
                }
              }
              
              newMessages = [...newMessages, sealedMsg];
              currentStreamText = "";
              assistantId = generateId();
              if (!hasAssistantSpeechStartedRef.current) {
                hasAssistantSpeechStartedRef.current = true;
                scheduleHideWorkspaceThinking();
              }
            } else if (data.type === "finish") {
              setIsLoading(false);
              if (hasAssistantSpeechStartedRef.current) scheduleHideWorkspaceThinking();
              else setShowWorkspaceThinking(false);
              thinkingHideScheduledRef.current = false;
              workspaceSocketRef.current = null;
              ws.close();
            } else if (data.type === "error") {
              const errorText = typeof data.content === "string" && data.content.trim()
                ? data.content
                : "⚠️ [Swarm Error]: backend returned an empty error payload. Please check backend logs.";
              console.error("Agent Swarm Error:", errorText);
              updateSession(currentSessionId, {
                messages: [...newMessages, { id: assistantId, role: "assistant", content: currentStreamText + `\n\n${errorText}`, name: "System Interface" }]
              });
              setIsLoading(false);
              if (hasAssistantSpeechStartedRef.current) scheduleHideWorkspaceThinking();
              else setShowWorkspaceThinking(false);
              thinkingHideScheduledRef.current = false;
              workspaceSocketRef.current = null;
              ws.close();
            }
          } catch (e) {
            console.error("WebSocket message parse error", e);
          }
        };

        ws.onerror = (error) => {
          console.error("FastAPI WebSocket error. Ensure Python backend is running.", error);
          updateSession(currentSessionId, {
            messages: [...newMessages, { id: assistantId, role: "assistant", content: `⚠️ [Connection Error]: Could not reach the Multi-Agent Engine WebSocket at 127.0.0.1:8000.\n\nPlease ensure your Python FastAPI backend is running.`, name: "System Interface" }]
          });
          setIsLoading(false);
          if (hasAssistantSpeechStartedRef.current) scheduleHideWorkspaceThinking();
          else setShowWorkspaceThinking(false);
          thinkingHideScheduledRef.current = false;
          workspaceSocketRef.current = null;
        };

        ws.onclose = () => {
          setIsLoading(false);
          if (hasAssistantSpeechStartedRef.current) scheduleHideWorkspaceThinking();
          else setShowWorkspaceThinking(false);
          thinkingHideScheduledRef.current = false;
          workspaceSocketRef.current = null;
        };

        return; // Skip the standard HTTP fallback below
      }

      // Normal Mode: Single Model Vercel AI SDK Fallback
      const abortController = new AbortController();
      chatAbortControllerRef.current = abortController;
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages,
          mode: "chat",
          model: currentSession?.model,
          apiKey: apiKey
        }),
        signal: abortController.signal,
      });

      if (!res.body) throw new Error("No response body from server.");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantText += chunk;
        if (assistantText.trim()) {
          hasAssistantSpeechStartedRef.current = true;
        }

        updateSession(currentSessionId, {
          messages: [...newMessages, { id: assistantId, role: "assistant", content: assistantText }]
        });
      }

      if (assistantText.trim() === "") {
        const isLocal = currentSession?.model?.startsWith("ollama/");
        assistantText = `⚠️ [API Error]: The model returned an empty response.\n\n${isLocal ? "This is a local model. Please ensure the **Ollama** app is running on your Mac and the model is downloaded." : "This usually occurs because your **API Key is invalid or has insufficient quota**.\n\nPlease check the ℹ️ Info menu in the top right to verify your setup."}`;
        updateSession(currentSessionId, {
          messages: [...newMessages, { id: assistantId, role: "assistant", content: assistantText }]
        });
      }

      const activeSession = sessions.find(s => s.id === currentSessionId);
      if (activeSession?.mode === "workspace" && !activeSession.guideline) {
        updateSession(currentSessionId, { guideline: assistantText });
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      const detail = err instanceof Error ? err.message : String(err);
      console.error(err);
      updateSession(currentSessionId, {
        messages: [...newMessages, { id: assistantId, role: "assistant", content: `⚠️ [Network Error]: Could not connect to the chat server.\nDetail: ${detail}` }]
      });
    } finally {
      setIsLoading(false);
      if (hasAssistantSpeechStartedRef.current) scheduleHideWorkspaceThinking();
      else setShowWorkspaceThinking(false);
      thinkingHideScheduledRef.current = false;
      chatAbortControllerRef.current = null;
    }
  };

  const handleConfigChange = (newConfig: AgentConfig) => {
    setAgentConfigs((prev) => {
      const next = prev.map((c) => (c.role === newConfig.role ? newConfig : c));
      updateSession(currentSessionId, { agentConfigs: next });
      return next;
    });
  };

  const updateAgentAtIndex = (index: number, updater: (agent: AgentConfig) => AgentConfig) => {
    setAgentConfigs((prev) => {
      const next = prev.map((agent, agentIndex) => agentIndex === index ? updater(agent) : agent);
      updateSession(currentSessionId, { agentConfigs: next });
      return next;
    });
  };

  useEffect(() => {
    if (!isNearBottomRef.current) return;
    scrollMessagesToBottom(isLoading ? "auto" : "smooth");
  }, [currentSession?.messages.length, showWorkspaceThinking, isLoading, stage]);

  useEffect(() => {
    document.title = "Rlong GPT";
  }, []);

  const executionIssues = currentSession?.mode === "workspace"
    ? getWorkspaceExecutionIssues(agentConfigs, apiKey || currentSession?.apiKey || "", currentSession?.model || "")
    : [];

  // Next.js SSR Hydration Safe Guard (Render identical empty shell on server)
  if (!mounted || !currentSession) {
    return <div className="h-screen w-screen bg-[#050505] text-white flex items-center justify-center font-sans tracking-widest text-sm uppercase opacity-50">Initializing Core...</div>;
  }

  return (
    <div 
      className="flex h-[100dvh] w-screen overflow-hidden text-[#ededed] font-sans relative z-0 p-3 sm:p-5 gap-3 sm:gap-5"
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsMouseOver(true)}
      onMouseLeave={() => setIsMouseOver(false)}
    >
      {/* Global Interactive Spotlight for Glassmorphism */}
      <div 
        className={`pointer-events-none fixed inset-0 z-[-5] transition-opacity duration-500 ${isMouseOver ? "opacity-100" : "opacity-0"}`}
        style={{
          background: `radial-gradient(500px circle at ${mousePos.x}px ${mousePos.y}px, rgba(255,255,255,0.08), transparent 40%)`
        }}
      />

      {/* Ultra-Realistic Space Photo */}
      <div className="stars-bg">
        <div className="stars-layer stars-1"></div>
        <div className="stars-layer stars-2"></div>
        <div className="stars-layer stars-3"></div>
        <div className="stars-layer stars-4"></div>
        <div className="stars-layer stars-5"></div>
      </div>
      <div className="meteor-shower">
        <div className="meteor meteor-1"></div>
        <div className="meteor meteor-2"></div>
        <div className="meteor meteor-3"></div>
      </div>

      {/* Info Modal */}
      {showHelp && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-md z-[100] flex items-center justify-center p-4">
          <div className="bg-[#1c1c1e]/95 border border-white/20 rounded-[32px] p-8 max-w-xl shadow-[0_0_50px_rgba(0,0,0,0.8)] space-y-5">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-xl font-bold text-white flex items-center gap-2"><Info className="w-5 h-5 text-indigo-400" /> API Key & Local Model Setup</h3>
              <button onClick={() => setShowHelp(false)} className="text-white/50 hover:text-white bg-white/5 hover:bg-white/10 p-2 rounded-full transition-colors"><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-4 text-[14.5px] text-white/80 leading-relaxed">
              <div className="bg-white/5 hover:bg-white/10 transition-colors p-5 rounded-[24px] border border-white/5">
                <strong className="text-white flex items-center gap-2 mb-2 text-[15px]"><span className="text-[#32ade6] font-black text-lg">1.</span> Free Gemini Pro (Google AI Studio)</strong>
                Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2 font-medium">Google AI Studio</a> to get a 100% free Gemini API Key. Click the 🔑 icon in the top right to paste it. Select <code>gemini-2.5-flash</code> from the model selector!
              </div>
              <div className="bg-white/5 hover:bg-white/10 transition-colors p-5 rounded-[24px] border border-white/5">
                <strong className="text-white flex items-center gap-2 mb-2 text-[15px]"><span className="text-indigo-400 font-black text-lg">2.</span> OpenAI API Key (GPT-4o)</strong>
                Paste your paid OpenAI API Key into the 🔑 field. It is securely saved in your browser storage.
              </div>
              <div className="bg-white/5 hover:bg-white/10 transition-colors p-5 rounded-[24px] border border-white/5">
                <strong className="text-white flex items-center gap-2 mb-2 text-[15px]"><span className="text-purple-400 font-black text-lg">3.</span> Local Ollama Models (No API Key)</strong>
                Simply select <code>ollama/llama3.2</code> or use <strong>Custom Model...</strong> to type <code>ollama/deepseek-coder</code>. Ensure your local Ollama app is running in the terminal!
              </div>
            </div>
            <button onClick={() => setShowHelp(false)} className="w-full bg-indigo-500 hover:bg-indigo-400 text-white py-3.5 rounded-[20px] transition-all shadow-[0_0_20px_rgba(99,102,241,0.4)] font-bold uppercase tracking-widest text-sm mt-4">Understood</button>
          </div>
        </div>
      )}

      {currentSession.mode === "workspace" && stage === "implementation" && guidelineContent && implementationReviewPending && (
        <div className="fixed inset-0 z-[150] flex items-center justify-center p-6">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-lg" />
          <div className="relative z-10 w-full max-w-5xl max-h-[85vh] bg-[#0d0d12]/92 border border-white/15 rounded-[32px] shadow-[0_0_90px_rgba(99,102,241,0.28)] overflow-hidden flex flex-col">
            <div className="px-8 py-5 border-b border-white/10 bg-indigo-500/10 flex items-center justify-between">
              <div>
                <h2 className="text-[22px] font-extrabold text-white tracking-tight flex items-center gap-2">
                  <Rocket className="w-5 h-5 text-indigo-300" />
                  Implementation Review
                </h2>
                <p className="text-white/50 text-[14px] mt-1">先在中央確認與修改 implementation，再按下 `Process` 直接開始 agents 協作。</p>
              </div>
              <button
              onClick={() => {
                  setStage("agent_config");
                  setIsPanelOpen(true);
                  setFocusSidebar(true);
                  setImplementationReviewPending(false);
                  updateSession(currentSessionId, { stage: "agent_config", guideline: guidelineContent, agentConfigs, implementationReviewPending: false });
                }}
                className="px-6 py-3 bg-indigo-500 hover:bg-indigo-400 text-white rounded-2xl font-extrabold text-[13px] tracking-[0.22em] uppercase shadow-[0_0_30px_rgba(99,102,241,0.38)]"
              >
                Process
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-8 space-y-4">
              <textarea
                value={guidelineContent}
                onChange={(e) => {
                  const next = e.target.value;
                  setGuidelineContent(next);
                  updateSession(currentSessionId, { guideline: next, stage: "implementation", implementationReviewPending: true });
                }}
                className="w-full min-h-[360px] resize-y bg-black/40 border border-white/10 rounded-[24px] p-5 text-[14px] leading-7 text-white/90 outline-none focus:border-indigo-400/50 transition-colors"
              />
              <div className="flex flex-col xl:flex-row gap-3">
                <input
                  className="flex-1 bg-white/5 border border-white/15 rounded-2xl px-4 py-3 text-[14px] text-white placeholder-white/35 outline-none focus:border-indigo-500/50 transition-colors"
                  placeholder="用 AI 修改 guideline，例如：補上 reviewer 驗收規範、改技術棧、調整功能拆分"
                  value={guidelineRefineInput}
                  onChange={(e) => setGuidelineRefineInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && guidelineRefineInput.trim() && !isRefiningGuideline) {
                      e.preventDefault();
                      refineGuidelineWithAI();
                    }
                  }}
                />
                <button
                  onClick={refineGuidelineWithAI}
                  disabled={!guidelineRefineInput.trim() || isRefiningGuideline}
                  className="px-5 py-3 bg-white/10 hover:bg-white/15 disabled:opacity-40 disabled:cursor-not-allowed rounded-2xl text-white/90 transition-colors flex items-center justify-center gap-2"
                >
                  <Wand2 className="w-4 h-4" />
                  {isRefiningGuideline ? "Refining..." : "AI 修改"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Floating Curved Left Sidebar */}
      <div
        className={`flex-shrink-0 bg-white/[0.01] backdrop-blur-[6px] border-[1.5px] border-white/20 flex flex-col z-20 shadow-[inset_0_0_20px_rgba(255,255,255,0.03),0_10px_30px_rgba(0,0,0,0.5)] relative transition-all duration-500 ease-out rounded-[32px] overflow-hidden ${isSidebarOpen ? "w-[260px] opacity-100" : "w-0 opacity-0 border-none shadow-none"}`}
      >
        <div className="h-[70px] flex items-center px-5 border-b border-white/5 shrink-0 gap-3">
          <div className="w-9 h-9 rounded-2xl shadow-[0_0_15px_rgba(255,255,255,0.1)] overflow-hidden border border-white/10 bg-white/10 p-1">
            <img src="/logo.svg" alt="Logo" className="w-full h-full object-contain" />
          </div>
          <span className="font-extrabold tracking-tight text-white text-[18px] drop-shadow-md whitespace-nowrap">Rlong GPT</span>
        </div>
        <div className="p-4 flex gap-2">
          <button
            onClick={() => createNewSession("chat")}
            className="flex-1 flex items-center justify-center gap-2 bg-white/5 hover:bg-white/15 border border-white/5 shadow-[0_2px_10px_rgba(0,0,0,0.2)] rounded-[20px] px-3 py-3 text-sm font-semibold transition-all hover:scale-105"
          >
            <MessageSquare className="w-4 h-4" /> Chat
          </button>
          <button
            onClick={() => createNewSession("workspace")}
            className="flex-1 flex items-center justify-center gap-2 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-400/20 shadow-[0_0_15px_rgba(99,102,241,0.2)] rounded-[20px] px-3 py-3 text-sm font-semibold transition-all hover:scale-105"
          >
            <Rocket className="w-4 h-4" /> Project
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1.5 scrollbar-hide">
          <div className="text-xs font-bold text-white/30 mb-3 px-2 pt-2 flex items-center gap-1.5 uppercase tracking-widest"><Clock className="w-3.5 h-3.5" /> History</div>
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center justify-between px-2 py-1.5 rounded-[20px] transition-all ${currentSessionId === s.id ? "bg-white/20 text-white shadow-md border border-white/10" : "text-white/50 hover:bg-white/10 hover:text-white/90"}`}
            >
              <button
                onClick={() => { setCurrentSessionId(s.id); setIsPanelOpen(false); }}
                className="flex-1 text-left truncate px-2 py-1.5 text-[14px] font-medium"
              >
                <span className="opacity-70 mr-1">{s.mode === "workspace" ? "🚀" : "💬"}</span> {s.title}
              </button>
              <button
                onClick={(e) => deleteSession(s.id, e)}
                className="opacity-0 group-hover:opacity-100 p-2 hover:bg-white/20 rounded-full text-red-400/80 hover:text-red-400 transition-all mr-1"
                title="Delete Chat"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Floating Curved Main Chat Area */}
      <div
        className={`flex-1 flex flex-col relative transition-all duration-500 ease-out z-10 bg-white/[0.01] backdrop-blur-[6px] border-[1.5px] border-white/20 rounded-[32px] overflow-hidden shadow-[inset_0_0_20px_rgba(255,255,255,0.03),0_10px_40px_rgba(0,0,0,0.6)] focus-lock-backdrop ${focusSidebar ? 'focus-sidebar-active' : ''}`}
        style={isPanelOpen ? { marginRight: `${panelWidth + 16}px` } : undefined}
      >

        {/* Header (Glass) */}
        <header className="h-[70px] flex items-center justify-between px-4 sm:px-6 border-b border-white/10 bg-black/10 shrink-0 sticky top-0 z-30 transition-all">
          <div className="flex items-center gap-3 font-semibold">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 bg-white/5 hover:bg-white/15 border border-white/10 rounded-full transition-colors text-white/60 hover:text-white"
            >
              {isSidebarOpen ? <PanelLeftClose className="w-4.5 h-4.5" /> : <PanelLeftOpen className="w-4.5 h-4.5" />}
            </button>
            <div className="w-px h-6 bg-white/10 mx-1 hidden sm:block"></div>
            {currentSession.mode === "workspace" ? (
              <div className="flex items-center gap-2"><Rocket className="w-5 h-5 text-indigo-400 drop-shadow-[0_0_8px_rgba(99,102,241,0.5)]" /> <span className="text-white drop-shadow-md tracking-wide">Workspace Mode</span></div>
            ) : (
              <div className="flex items-center gap-2"><MessageSquare className="w-5 h-5 text-[#32ade6] drop-shadow-[0_0_8px_rgba(50,173,230,0.5)]" /> <span className="text-white drop-shadow-md tracking-wide">Normal Chat</span></div>
            )}
          </div>

          <div className="flex gap-3 items-center">
            <div className="flex sm:gap-3 gap-2 items-center">
              <button
                onClick={() => setShowHelp(true)}
                className="w-[36px] h-[36px] flex items-center justify-center bg-white/10 hover:bg-white/20 border border-white/10 rounded-full text-white/70 hover:text-white transition-all shadow-[0_2px_10px_rgba(0,0,0,0.2)] ml-1"
                title="How to setup models"
              >
                <Info className="w-4.5 h-4.5" />
              </button>

              {!(currentSession.mode === "workspace"
                ? (agentConfigs.find((a) => a.role === "PM")?.model || currentSession.model || "").startsWith("ollama/")
                : (currentSession.model || "").startsWith("ollama/")) && (
                <div className="hidden sm:flex items-center bg-white/5 px-3 py-2 rounded-[18px] border border-white/10 text-sm overflow-hidden focus-within:border-indigo-500 focus-within:bg-black/40 focus-within:shadow-[0_0_20px_rgba(99,102,241,0.2)] transition-all">
                  <KeyRound className="w-4 h-4 text-white/50 mx-1" />
                  <input
                    type="password"
                    value={apiKey}
                    placeholder="API Key..."
                    onChange={(e) => {
                      setApiKey(e.target.value);
                      updateSession(currentSessionId, { apiKey: e.target.value });
                    }}
                    className="w-[200px] bg-transparent text-white/85 placeholder-white/35 outline-none border-0 shadow-none"
                  />
                </div>
              )}
              <div className="flex items-center gap-1 bg-white/5 hover:bg-white/10 px-4 py-2 rounded-[18px] border border-white/10 text-sm shadow-sm transition-all focus-within:border-indigo-500 focus-within:bg-black/40 focus-within:shadow-[0_0_20px_rgba(99,102,241,0.2)]">
                <select
                  value={isCustomModel ? "custom" : (currentSession.mode === "workspace"
                    ? (agentConfigs.find((a) => a.role === "PM")?.model || currentSession.model || "ollama/deepseek-r1:32b")
                    : currentSession.model)}
                  onChange={(e) => {
                    const newVal = e.target.value;
                    if (newVal === "custom") {
                      setIsCustomModel(true);
                      const customVal = customModelId || "ollama/mistral";
                      updateSession(currentSessionId, { model: customVal });
                      if (currentSession.mode === "workspace") {
                        setAgentConfigs(configs => {
                          const withPm = ensureWorkspacePmAgent(configs, customVal, apiKey || currentSession.apiKey || "");
                          updateSession(currentSessionId, { agentConfigs: withPm });
                          return withPm;
                        });
                      }
                    } else {
                      setIsCustomModel(false);
                      updateSession(currentSessionId, { model: newVal });
                      localStorage.setItem(LAST_MODEL_STORAGE_KEY, newVal);
                      if (currentSession.mode === "workspace") {
                        setAgentConfigs(configs => {
                          const withPm = ensureWorkspacePmAgent(configs, newVal, apiKey || currentSession.apiKey || "");
                          const next = withPm.map(c => c.role === "PM" ? { ...c, model: newVal } : c);
                          updateSession(currentSessionId, { agentConfigs: next });
                          return next;
                        });
                      }
                    }
                  }}
                  className="bg-transparent outline-none cursor-pointer appearance-none text-white font-semibold min-w-[140px]"
                >
                  <option value="gpt-4o" className="text-black">GPT-4o</option>
                  <option value="claude-3-5-sonnet-20240620" className="text-black">Claude 3.5</option>
                  <option value="gemini-2.5-flash" className="text-black">Gemini 2.5 Flash</option>
                  <option value="gemini-2.5-pro" className="text-black">Gemini 2.5 Pro</option>
                  <option disabled className="text-black">──────────</option>
                  <option value="ollama/deepseek-r1:32b" className="text-black">DeepSeek R1 32B (Local)</option>
                  <option value="ollama/llama3.2" className="text-black">Llama 3.2 (Local)</option>
                  <option value="ollama/qwen2.5" className="text-black">Qwen 2.5 (Local)</option>
                  <option value="custom" className="text-black">Custom Model...</option>
                </select>
                {!isCustomModel && <ChevronDown className="w-4 h-4 text-white/40 shrink-0 pointer-events-none" />}

                {isCustomModel && (
                  <input
                    type="text"
                    placeholder="Model tag..."
                    value={customModelId}
                    onChange={(e) => {
                      const val = e.target.value;
                      setCustomModelId(val);
                      updateSession(currentSessionId, { model: val });
                      localStorage.setItem(LAST_MODEL_STORAGE_KEY, val);
                      if (currentSession.mode === "workspace") {
                        setAgentConfigs(configs => {
                          const withPm = ensureWorkspacePmAgent(configs, val, apiKey || currentSession.apiKey || "");
                          const next = withPm.map(c => c.role === "PM" ? { ...c, model: val } : c);
                          updateSession(currentSessionId, { agentConfigs: next });
                          return next;
                        });
                      }
                    }}
                    className="bg-black/50 border border-indigo-500 text-white text-[13px] px-3 py-1.5 rounded-xl outline-none w-[140px] ml-3 shadow-[0_0_15px_rgba(99,102,241,0.2)] font-mono"
                  />
                )}
              </div>
            </div>

            {currentSession.mode === "workspace" && (stage === 'agent_config' || stage === 'execution') && (
              <button
                onClick={() => setIsPanelOpen(!isPanelOpen)}
                className={`rounded-[18px] bg-white/10 border border-white/10 hover:bg-white/20 transition-all p-2.5 shadow-md hover:scale-105 ml-2 ${isPanelOpen ? 'text-indigo-400 bg-indigo-500/20' : 'text-white'}`}
              >
                <LayoutPanelLeft className="w-5 h-5" />
              </button>
            )}
          </div>
        </header>

        {/* Chat Messages */}
        <div
          ref={messagesScrollRef}
          onScroll={handleMessagesScroll}
          className="flex-1 overflow-y-auto w-full px-5 sm:px-8 relative z-20 scrollbar-hide"
        >
          <div className="max-w-4xl mx-auto py-10 space-y-8 min-h-full pb-8">
            {currentSession.mode === "workspace" && stage === "execution" && executionQueue.length > 0 && (
              <div className="sticky top-4 z-20 mb-3 flex justify-start">
                <div className={`transition-all duration-300 ${executionBannerCollapsed ? "w-auto" : "w-full max-w-[760px]"}`}>
                  {executionBannerCollapsed ? (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setExecutionBannerCollapsed(false)}
                        className="flex items-center gap-2 rounded-full border border-white/10 bg-black/45 px-3 py-2 backdrop-blur-xl shadow-[0_8px_24px_rgba(0,0,0,0.25)] text-left"
                      >
                        <div className={`h-2 w-2 rounded-full ${isLoading ? "bg-emerald-400 animate-pulse shadow-[0_0_12px_rgba(74,222,128,0.8)]" : "bg-white/25"}`} />
                        <span className="text-[11px] font-black uppercase tracking-[0.22em] text-white/45">Queue</span>
                        <span className="text-[12px] text-white/82 font-semibold">{activeExecutionRole || "Waiting"}</span>
                        <span className="text-[11px] text-white/35">◀</span>
                      </button>
                      {isLoading && (
                        <button
                          type="button"
                          onClick={stopCurrentResponse}
                          className="flex items-center gap-2 rounded-full border border-red-400/25 bg-red-500/15 px-3 py-2 text-[11px] font-black uppercase tracking-[0.18em] text-red-100 shadow-[0_8px_24px_rgba(127,29,29,0.25)] transition-colors hover:bg-red-500/25"
                        >
                          <Square className="h-3.5 w-3.5 fill-current" />
                          停止
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-[22px] border border-white/10 bg-[linear-gradient(180deg,rgba(10,16,24,0.82),rgba(7,10,16,0.62))] px-4 py-3 backdrop-blur-xl shadow-[0_14px_34px_rgba(0,0,0,0.28)]">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <div className={`h-2 w-2 rounded-full ${isLoading ? "bg-emerald-400 animate-pulse shadow-[0_0_10px_rgba(74,222,128,0.75)]" : "bg-white/25"}`} />
                            <span className="text-[10px] font-black uppercase tracking-[0.24em] text-white/38">Execution Queue</span>
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                            <span className="text-[14px] font-semibold text-white/88">
                              {activeExecutionRole ? `目前執行：${activeExecutionRole}` : "等待下一位 agent"}
                            </span>
                            <span className="rounded-full border border-white/8 bg-white/[0.04] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em] text-white/48">
                              {isLoading ? "Running" : "Waiting"}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-[12px] text-white/50 font-medium tabular-nums">
                            {completedExecutionCount} / {executionQueue.length}
                          </div>
                          {isLoading && (
                            <button
                              type="button"
                              onClick={stopCurrentResponse}
                              className="rounded-full border border-red-400/25 bg-red-500/15 px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-red-100 shadow-[0_8px_24px_rgba(127,29,29,0.22)] transition-colors hover:bg-red-500/25"
                            >
                              停止執行
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() => setExecutionBannerCollapsed(true)}
                            className="rounded-full border border-white/8 bg-white/[0.04] px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-white/48 hover:bg-white/[0.08] hover:text-white/75"
                          >
                            Hide
                          </button>
                        </div>
                      </div>

                      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(16,185,129,0.9),rgba(99,102,241,0.9))] transition-all duration-500"
                          style={{ width: `${Math.max((completedExecutionCount / executionQueue.length) * 100, activeExecutionRole ? 8 : 0)}%` }}
                        />
                      </div>

                      <div className="mt-3 flex flex-wrap gap-2">
                        {executionQueue.map((role, index) => {
                          const isDone = index < completedExecutionCount;
                          const isActive = index === activeExecutionIndex;
                          const color = getAgentColor(role);
                          return (
                            <div
                              key={`${role}-${index}`}
                              className={`rounded-full border px-3 py-1.5 text-[11px] font-semibold tracking-[0.02em] transition-all ${
                                isActive ? "shadow-[0_0_18px_rgba(255,255,255,0.08)] -translate-y-[1px]" : ""
                              }`}
                              style={{
                                backgroundColor: isDone ? "rgba(16,185,129,0.14)" : (isActive ? color.bg : "rgba(255,255,255,0.025)"),
                                borderColor: isDone ? "rgba(16,185,129,0.25)" : (isActive ? color.border : "rgba(255,255,255,0.07)"),
                                color: isDone ? "#86efac" : (isActive ? color.text : "rgba(255,255,255,0.62)"),
                              }}
                            >
                              {isDone ? "✓ " : isActive ? "→ " : ""}
                              {role}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {currentSession.messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center mt-[15vh] space-y-6 opacity-95">
                {currentSession.mode === "workspace" ? (
                  <><div className="w-[96px] h-[96px] bg-black/20 border border-white/10 backdrop-blur-2xl rounded-[32px] shadow-[0_0_50px_rgba(99,102,241,0.4)] flex items-center justify-center p-3">
                    <img src="/logo.svg" alt="" className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]" />
                  </div>
                    <h2 className="text-[32px] font-extrabold tracking-tight text-white drop-shadow-lg">Architect PM Agent</h2>
                    <p className="max-w-lg text-white/50 text-[16px] font-medium leading-relaxed">Describe your application vision. I will analyze the requirements and draft an implementation guideline for the engineering agents.</p></>
                ) : (
                  <><div className="w-[96px] h-[96px] bg-black/20 border border-white/10 backdrop-blur-2xl rounded-[32px] shadow-[0_0_50px_rgba(10,132,255,0.4)] flex items-center justify-center p-3">
                    <img src="/logo.svg" alt="" className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]" />
                  </div>
                    <h2 className="text-[32px] font-extrabold tracking-tight text-white drop-shadow-lg">How can I help you today?</h2></>
                )}
              </div>
            )}

            {currentSession.messages.map((m, idx) => (
              <div key={m.id} className={`flex gap-5 group ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {m.role === 'assistant' && (
                  <div className="w-[38px] h-[38px] rounded-[14px] bg-black/30 backdrop-blur-xl shrink-0 flex items-center justify-center shadow-[0_5px_15px_rgba(0,0,0,0.5)] border border-white/10 mt-1 p-1.5">
                    <img src="/logo.svg" alt="Bot" className="w-full h-full object-contain drop-shadow-md" />
                  </div>
                )}

                <div className={`max-w-[85%] prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10 break-words relative ${m.role === 'user' ? 'bg-[#0a84ff] text-white px-6 py-4 rounded-[28px] rounded-br-[8px] shadow-[0_10px_30px_rgba(10,132,255,0.25)] backdrop-blur-md border border-[#32ade6]/30 w-fit ml-auto' : 'flex-1 backdrop-blur-[6px] px-6 py-4 rounded-[28px] rounded-bl-[8px] shadow-[0_5px_20px_rgba(0,0,0,0.2)] transition-all duration-500'}`}
                  style={m.role === 'assistant' && currentSession.mode === 'workspace' ? { backgroundColor: getAgentColor(m.name || 'PM').bg, borderLeft: `3px solid ${getAgentColor(m.name || 'PM').border}`, border: `1px solid ${getAgentColor(m.name || 'PM').border}` } : m.role === 'assistant' ? { backgroundColor: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.15)' } : undefined}
                >
                  {m.role === 'assistant' && (() => {
                      const agentName = m.name || (currentSession.mode === 'workspace' ? 'PM' : 'Rlong GPT');
                      const color = currentSession.mode === 'workspace' ? getAgentColor(agentName) : null;
                      return (
                        <div className="font-bold text-[11px] mb-2.5 uppercase tracking-widest flex items-center gap-1.5" style={{ color: color?.text || 'rgba(255,255,255,0.4)' }}>
                          {currentSession.mode === 'workspace' ? <Rocket className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                          {agentName}
                        </div>
                      );
                    })()}

                  {/* Inline edit mode for user messages */}
                  {m.role === 'user' && editingMsgId === m.id ? (
                    <div className="flex flex-col gap-2 min-w-[220px]">
                      <textarea
                        className="bg-[#0a84ff]/10 border border-[#0a84ff]/40 rounded-2xl px-4 py-3 text-[15px] font-medium text-white outline-none resize-none w-full min-h-[70px] focus:bg-[#0a84ff]/15 focus:border-[#0a84ff]/60 transition-all duration-300 shadow-[0_4px_20px_rgba(10,132,255,0.15)] placeholder-white/50"
                        value={editingContent}
                        autoFocus
                        onChange={e => setEditingContent(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Apply edit: update messages list, truncate after this message, and re-submit
                            const newMsgs = currentSession.messages.slice(0, idx);
                            updateSession(currentSessionId, { messages: newMsgs });
                            setEditingMsgId(null);
                            setTimeout(() => submitChat(undefined, editingContent), 0);
                          }
                          if (e.key === 'Escape') setEditingMsgId(null);
                        }}
                      />
                      <div className="flex gap-2 justify-end text-[12.5px] mt-1">
                        <button
                          className="px-3.5 py-1.5 rounded-xl bg-white/5 hover:bg-white/15 text-white/70 transition-colors"
                          onClick={() => setEditingMsgId(null)}
                        >Cancel</button>
                        <button
                          className="px-4 py-1.5 rounded-xl bg-[#0a84ff] hover:bg-blue-400 text-white font-semibold shadow-[0_2px_10px_rgba(10,132,255,0.3)] transition-all hover:-translate-y-[1px]"
                          onClick={() => {
                            const newMsgs = currentSession.messages.slice(0, idx);
                            updateSession(currentSessionId, { messages: newMsgs });
                            setEditingMsgId(null);
                            setTimeout(() => submitChat(undefined, editingContent), 0);
                          }}
                        >Update & Send</button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="whitespace-pre-wrap text-[15.5px] font-medium leading-[1.65] drop-shadow-sm">
                        {(() => {
                          let display = m.content;
                          const isStreamingThisMessage = isLoading && m.role === 'assistant' && idx === currentSession.messages.length - 1;
                          
                          // Artificially enforce <think> state during early fragmented stream tokens or empty start
                          const contentTrim = display.trim();
                          const isFragement = contentTrim === '' || contentTrim === '<' || contentTrim === '<t' || contentTrim === '<th' || contentTrim === '<thi' || contentTrim === '<thin' || contentTrim === '<think';
                          if (isFragement && isLoading && m.id === currentSession.messages[currentSession.messages.length - 1]?.id) {
                             display = "<think>";
                          }

                          // Compress excessive newlines that cause huge vertical gaps (skip aggressive normalization while token streaming)
                          if (!isStreamingThisMessage) {
                            display = normalizeDisplayText(display);
                          } else {
                            display = display.replace(/\r\n/g, "\n");
                          }

                          // Obscure raw JSON tool calls during streaming
                          display = display.replace(/```(?:json)?\s*\{\s*"name"\s*:[^`]*```/g, "\n⚙️ Processing tools...\n");
                          display = display.replace(/^\{\s*"name"\s*:[\s\S]{0,500}?\}/gm, "\n⚙️ Processing tools...\n");
                          
                          // Phase 3 collapsible logic
                          const containsCodeBlock = /```/.test(display);
                          const isAgentBubble = currentSession.mode === "workspace" && m.role === 'assistant' && stage === 'execution' && !isStreamingThisMessage;
                          const isExpanded = expandedMsgs.has(m.id);

                          // Parse <think> blocks
                          const parts = display.split(/(<think>|<\/think>)/);
                          let inThink = false;
                          const elements = parts.map((part, i) => {
                            if (part === '<think>') { inThink = true; return null; }
                            if (part === '</think>') { inThink = false; return null; }
                            if (!part.trim() && !inThink) return null;
                            if (inThink) {
                              const isActive = i === parts.length - 1 && isLoading && m.id === currentSession.messages[currentSession.messages.length - 1]?.id;
                              return (
                                <details
                                  key={`think-${i}`}
                                  className={`group mb-3 text-white/50 bg-black/20 rounded-xl p-3 text-[14px] shadow-inner transition-all duration-300 ${isActive ? 'thinking-border-active' : 'border border-white/5'}`}
                                  open={currentSession.mode === "workspace" ? isActive : false}
                                >
                                  <summary className="cursor-pointer hover:text-white/80 transition-colors select-none font-semibold flex items-center justify-between h-[24px] list-none">
                                    {isActive ? (
                                      <span className="flex items-center gap-2 text-[#ff4d4d] drop-shadow-[0_0_8px_rgba(255,0,0,0.8)] font-mono">
                                        Thinking<span className="think-anim"></span>
                                      </span>
                                    ) : (
                                       <span className="flex items-center gap-2">
                                         Thinking Process
                                       </span>
                                    )}
                                    <span className="text-[10px] opacity-40 transition-transform duration-300 group-open:rotate-180">▼</span>
                                  </summary>
                                  <div className="mt-3 whitespace-pre-wrap leading-relaxed opacity-80 pl-1 border-l-2 border-white/5">{part.trim() || "..."}</div>
                                </details>
                              );
                            }
                            // Render Markdown with custom code block details wrapper
                            if (isStreamingThisMessage) {
                              return (
                                <div key={i} className="whitespace-pre-wrap text-[15.5px] font-medium leading-[1.65]">
                                  {part}
                                </div>
                              );
                            }
                            return (
                              <div key={i} data-color-mode="dark" className="markdown-transparent-bg">
                                <MarkdownPreview 
                                  source={part} 
                                  style={{ backgroundColor: 'transparent', color: 'inherit', fontSize: '15.5px', fontFamily: 'inherit' }}
                                  components={{
                                    pre: ({ children, ...props }: HTMLAttributes<HTMLPreElement>) => {
                                      return (
                                        <details className="my-3 border border-white/20 rounded-xl bg-black/60 overflow-hidden shadow-xl" open>
                                          <summary className="p-2.5 bg-white/10 cursor-pointer hover:bg-white/20 text-[13px] text-white/80 font-mono tracking-wide select-none">
                                            👨‍💻 Source Code Snippet
                                          </summary>
                                          <pre {...props} className="bg-transparent m-0 p-4 overflow-x-auto text-[14px]">
                                            {children}
                                          </pre>
                                        </details>
                                      );
                                    }
                                  }}
                                />
                              </div>
                            );
                          }).filter(Boolean);

                          return (
                            <div className="flex flex-col">
                              <div className={`transition-all duration-500 overflow-hidden relative ${isAgentBubble && !isExpanded ? 'max-h-[140px]' : ''}`}>
                                {elements}
                                {isAgentBubble && !isExpanded && !containsCodeBlock && (
                                  <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-[#1a1a1e] to-transparent pointer-events-none" style={{ mixBlendMode: 'multiply' }}></div>
                                )}
                              </div>
                              {isAgentBubble && (
                                <button
                                  className="mt-2 self-start text-[12px] font-semibold text-white/50 hover:text-white transition-colors uppercase tracking-wider bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-lg border border-white/5 flex items-center gap-1.5"
                                  onClick={() => {
                                    setExpandedMsgs(prev => {
                                      const next = new Set(prev);
                                      if (next.has(m.id)) next.delete(m.id);
                                      else next.add(m.id);
                                      return next;
                                    });
                                  }}
                                >
                                  {isExpanded ? "▲ Collapse Message" : "▼ Expand Full Message"}
                                </button>
                              )}
                            </div>
                          );
                        })()}
                      </div>
                      {/* Show edit pencil on user messages on hover */}
                      {m.role === 'user' && !isLoading && (
                        <button
                          className="absolute -left-9 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white/60 hover:text-white"
                          title="Edit message"
                          onClick={() => { setEditingMsgId(m.id); setEditingContent(m.content); }}
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
            {currentSession.mode === "workspace" && showWorkspaceThinking && (
              <div className="flex gap-5">
                <div className="w-[38px] h-[38px] rounded-[14px] bg-black/30 backdrop-blur-xl shrink-0 flex items-center justify-center shadow-lg border border-white/10 mt-1 p-1.5 animate-pulse">
                  <img src="/logo.svg" alt="Bot" className="w-full h-full object-contain drop-shadow-[0_0_8px_rgba(255,255,255,0.4)]" />
                </div>
                <div className="bg-black/40 backdrop-blur-xl px-5 py-4 rounded-[24px] rounded-bl-[8px] transition-all duration-500 overflow-visible relative thinking-border-active min-w-[220px]">
                  <div className="text-[#ff4d4d] drop-shadow-[0_0_10px_rgba(255,50,50,0.8)] font-mono font-bold tracking-widest text-[14px] flex items-center gap-1 mb-3">
                    Thinking<span className="think-anim"></span>
                  </div>
                  <details className="group text-white/60 bg-black/20 rounded-xl p-3 text-[13px] border border-white/5" open>
                    <summary className="cursor-pointer select-none font-semibold flex items-center justify-between list-none">
                      <span>Thinking Process</span>
                      <span className="text-[10px] opacity-40 transition-transform duration-300 group-open:rotate-180">▼</span>
                    </summary>
                    <div className="mt-2 whitespace-pre-wrap leading-relaxed border-l-2 border-white/5 pl-3">
                      {thinkingContent.trim() || "..."}
                    </div>
                  </details>
                </div>
              </div>
            )}
            {isLoading && currentSession.mode !== "workspace" && currentSession.messages[currentSession.messages.length - 1]?.role === "user" && (
              <div className="flex gap-5">
                <div className="w-[38px] h-[38px] rounded-[14px] bg-black/30 backdrop-blur-xl shrink-0 flex items-center justify-center shadow-lg border border-white/10 mt-1 p-1.5 animate-pulse">
                  <img src="/logo.svg" alt="Bot" className="w-full h-full object-contain drop-shadow-[0_0_8px_rgba(255,255,255,0.4)]" />
                </div>
                <div className="bg-black/40 backdrop-blur-xl px-7 py-4 rounded-[24px] rounded-bl-[8px] flex items-center justify-center transition-all duration-500 overflow-visible relative thinking-border-active min-w-[120px] min-h-[50px]">
                  <span className="text-[#ff4d4d] drop-shadow-[0_0_10px_rgba(255,50,50,0.8)] font-mono font-bold tracking-widest text-[14px] flex items-center gap-1">Thinking<span className="think-anim"></span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="w-full shrink-0 bg-transparent pt-2 pb-6 px-6 z-30">
          <div className="max-w-4xl mx-auto relative group flex flex-col gap-2">
            <form onSubmit={submitChat} className="relative w-full">
              <textarea
                className="w-full resize-none bg-white/[0.02] backdrop-blur-[8px] border-[1.5px] border-white/25 hover:border-white/40 rounded-[32px] pl-7 pr-16 py-5 text-[15.5px] placeholder-white/40 scrollbar-hide focus-animated outline-none shadow-[inset_0_0_20px_rgba(255,255,255,0.03),0_20px_50px_rgba(0,0,0,0.6)] font-medium text-white transition-all duration-500 ease-out"
                placeholder={
                  currentSession.mode === "workspace"
                    ? (stage === "discovery"
                      ? "Describe your project idea..."
                      : stage === "implementation"
                        ? "請先在中央確認 implementation 草案，然後按 Process"
                        : stage === "agent_config"
                          ? "請先在右側確認 agents，然後按 Start Execution"
                          : "Send a message to the agent team...")
                    : "Message AI..."
                }
                value={localInput}
                rows={Math.min(Math.max((localInput || '').split('\n').length, 1), 7)}
                onChange={(e) => setLocalInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.keyCode === 229) return;
                  if (e.key === "Enter" && !e.shiftKey) {
                    if (e.nativeEvent.isComposing) return;
                    e.preventDefault();
                    submitChat();
                  }
                }}
              />
              <button
                type={isLoading ? "button" : "submit"}
                onClick={isLoading ? stopCurrentResponse : undefined}
                disabled={!isLoading && (!localInput.trim() || (currentSession.mode === "workspace" && isPreExecutionStage(stage)))}
                aria-label={isLoading ? "停止回覆" : "送出訊息"}
                title={isLoading ? "停止回覆" : "送出訊息"}
                className={`absolute right-4 bottom-4 p-3 rounded-[20px] transition-all duration-300 ${isLoading
                  ? "bg-red-500 text-white shadow-[0_0_25px_rgba(255,77,77,0.55)] hover:scale-110"
                  : localInput.trim()
                    ? (currentSession.mode === 'workspace' ? 'bg-indigo-500 text-white shadow-[0_0_25px_rgba(99,102,241,0.6)] hover:scale-110' : 'bg-white text-black shadow-[0_0_25px_rgba(255,255,255,0.6)] hover:scale-110')
                    : 'bg-white/10 text-white/30'
                }`}
              >
                {isLoading ? <Square className="w-[18px] h-[18px] fill-current" /> : <Send className="w-[20px] h-[20px]" />}
              </button>
            </form>
            {currentSession.mode === "workspace" && isPreExecutionStage(stage) && (
              <div className="px-4 text-[12px] text-white/45 text-center">
                {stage === "implementation"
                  ? "現在是 implementation 草案確認階段，按 Process 就會直接開始 agents 協作。"
                  : "現在是 agent 協作階段，你仍可在右側修改角色與 prompt。"}
              </div>
            )}
            
            {/* Action Buttons — Regenerate only */}
            {currentSession.messages.length > 0 && currentSession.messages[currentSession.messages.length - 1].role === "assistant" && (
              <div className="flex gap-2 px-4 justify-end">
                <button
                  type="button"
                  onClick={() => {
                    const lastUser = currentSession.messages.filter(m => m.role === "user").pop();
                    if (lastUser) {
                      setLocalInput(lastUser.content);
                      setTimeout(() => submitChat(), 0);
                    }
                  }}
                  className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-xl text-xs font-semibold text-white/70 transition-colors flex items-center gap-1.5 border border-white/5"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Regenerate
                </button>
              </div>
            )}
          </div>
          <div className="text-center mt-5 text-[11px] text-white/30 tracking-widest font-extrabold uppercase drop-shadow-md">
            Segora © 2026 All Right Reserved.
          </div>
        </div>
      </div>

      {/* Right Artifact Panel */}
      <div
        className={`fixed top-4 bottom-4 right-4 bg-white/[0.01] backdrop-blur-[6px] border-[1.5px] border-white/20 rounded-[32px] z-[40] flex flex-col shadow-[inset_0_0_20px_rgba(255,255,255,0.03),-20px_0_50px_rgba(0,0,0,0.6)] transition-all duration-500 ease-out overflow-hidden ${isPanelOpen ? "translate-x-0 opacity-100" : "translate-x-full opacity-0 border-none shadow-none"}`}
        style={{ width: isPanelOpen ? `${panelWidth}px` : "0px" }}
      >
        {isPanelOpen && (
          <div
            onPointerDown={beginPanelResize}
            className="absolute left-0 top-0 bottom-0 w-2 cursor-col-resize z-50 hover:bg-indigo-400/20 transition-colors"
            title="Drag to resize panel"
          >
            <div className="absolute left-1 top-1/2 -translate-y-1/2 h-24 w-px bg-white/15" />
          </div>
        )}
        <div className="flex items-center justify-between p-6 border-b border-white/10 bg-black/10">
          <h2 className="font-exrabold text-[17px] tracking-tight text-white">Project Configuration</h2>
          <button onClick={() => setIsPanelOpen(false)} className="text-white/50 hover:text-white p-2 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          <div className="space-y-4">
            <h3 className="text-[12px] font-black text-white/40 uppercase tracking-widest pl-1">Agent Team Configuration</h3>
            <div className="bg-black/20 border border-white/10 rounded-[24px] p-4 space-y-3 shadow-inner backdrop-blur-xl">
              {agentConfigs.map((config, index) => {
                const color = getAgentColor(config.role);
                return (
                  <div key={`${config.role}-${index}`} className="flex flex-col gap-2 py-3 border-b border-white/5 last:border-0 rounded-2xl transition-colors px-2" style={{ borderLeft: `3px solid ${color.border}` }}>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-3">
                        <div className="w-7 h-7 rounded-lg flex items-center justify-center border p-1.5 shadow-sm" style={{ backgroundColor: color.bg, borderColor: color.border }}>
                          <Bot className="w-full h-full" style={{ color: color.text }} />
                        </div>
                        <input
                          value={config.role}
                          onChange={(e) => updateAgentAtIndex(index, (agent) => ({ ...agent, role: e.target.value }))}
                          className="bg-transparent border-b border-white/10 focus:border-white/30 outline-none text-[14px] font-bold min-w-[96px]"
                          style={{ color: color.text }}
                        />
                      </div>
                      <div className="flex items-center gap-1">
                        <select
                          value={config.model}
                          onChange={(e) => updateAgentAtIndex(index, (agent) => ({ ...agent, model: e.target.value }))}
                          className="bg-black/50 border border-white/10 text-white text-[11px] px-2 py-1 rounded-lg outline-none max-w-[120px]"
                        >
                          <option value="gpt-4o" className="text-black">GPT-4o</option>
                          <option value="gemini-2.5-flash" className="text-black">Gemini Flash</option>
                          <option value="gemini-2.5-pro" className="text-black">Gemini Pro</option>
                          <option value="ollama/deepseek-r1:32b" className="text-black">DeepSeek (Local)</option>
                          <option value="ollama/llama3.2" className="text-black">Llama 3.2 (Local)</option>
                        </select>
                        <button
                          title="Delete agent"
                          onClick={() => { setAgentConfigs(prev => { const next = prev.filter((_, itemIndex) => itemIndex !== index); updateSession(currentSessionId, { agentConfigs: next }); return next; }); }}
                          className="p-1 rounded-full hover:bg-red-500/20 text-white/30 hover:text-red-400 transition-colors"
                        ><Trash2 className="w-3 h-3" /></button>
                      </div>
                    </div>
                    <div className="relative">
                      <textarea
                        placeholder={`${config.role} system prompt...`}
                        value={config.prompt || ""}
                        onChange={(e) => updateAgentAtIndex(index, (agent) => ({ ...agent, prompt: e.target.value }))}
                        className="w-full h-[112px] bg-black/40 border border-white/10 hover:border-white/20 rounded-[12px] px-3 py-2 text-[12px] text-white/80 placeholder-white/30 resize-none outline-none focus:border-indigo-400/50 transition-colors pr-8"
                      />
                      <button
                        title="AI Generate Prompt"
                        onClick={() => generateAgentPrompt(config, index)}
                        className="absolute right-2 top-2 p-1 rounded-md hover:bg-indigo-500/20 text-white/30 hover:text-indigo-400 transition-colors"
                      ><Sparkles className="w-3.5 h-3.5" /></button>
                    </div>
                    {!isLocalModel(config.model) && (
                      <input
                        type="password"
                        value={config.apiKey ?? apiKey}
                        onChange={(e) => updateAgentAtIndex(index, (agent) => ({ ...agent, apiKey: e.target.value }))}
                        placeholder={`${config.role} API Key`}
                        className="w-full bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-[12px] text-white/80 placeholder-white/35 outline-none focus:border-indigo-400/50"
                      />
                    )}
                  </div>
                );
              })}
              <div className="rounded-2xl border border-dashed border-white/15 p-3 space-y-3">
                <input
                  value={pendingAgentRole}
                  onChange={(e) => setPendingAgentRole(e.target.value)}
                  placeholder="New agent role name"
                  className="w-full bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-[13px] text-white outline-none focus:border-indigo-400/50"
                />
                <button
                  onClick={() => {
                    if (!pendingAgentRole.trim()) return;
                      const pmModel = agentConfigs.find(c => c.role === 'PM')?.model || 'gpt-4o';
                      setAgentConfigs(prev => {
                        const next = [...prev, { role: pendingAgentRole.trim(), model: pmModel, prompt: '', apiKey }];
                        updateSession(currentSessionId, { agentConfigs: next });
                        return next;
                      });
                    setPendingAgentRole("");
                  }}
                  className="w-full py-3 text-[13px] font-semibold flex items-center justify-center gap-2 transition-colors hover:bg-white/5 rounded-xl text-white/70"
                ><Plus className="w-4 h-4" /> Add Agent</button>
              </div>
            </div>
          </div>

          {/* Start Execution Button */}
          {stage === 'agent_config' && (
            <div className="space-y-3">
              <button
                onClick={() => {
                  if (executionIssues.length > 0) return;
                  const executionModel = getWorkspaceExecutionModel(agentConfigs, currentSession.model || apiKey || "ollama/qwen2.5");
                  const finalAgentConfigs = agentConfigs.map((agent) => {
                    const agentModel = agent.model || executionModel;
                    return {
                      ...agent,
                      model: agentModel,
                      apiKey: isLocalModel(agentModel) ? "" : (agent.apiKey || apiKey),
                    };
                  });
                  setFocusSidebar(false);
                  setIsPanelOpen(false);
                  setStage('execution');
                  setAgentConfigs(finalAgentConfigs);
                  setImplementationReviewPending(false);
                  updateSession(currentSessionId, { stage: 'execution', guideline: guidelineContent, agentConfigs: finalAgentConfigs, implementationReviewPending: false });
                  submitChat(undefined, '[SYSTEM] The user has approved the implementation guideline and agent team. Begin multi-agent execution now and follow the prompts strictly. Work only inside the isolated agent workspace folder, not in the currently running app repo. Treat that isolated workspace as the project root for this generated deliverable. If a role is asked to create multiple files in one work package, complete all of them in the same round instead of one file per round.', finalAgentConfigs, 'execution');
                }}
                disabled={executionIssues.length > 0}
                className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-white/10 disabled:text-white/35 disabled:cursor-not-allowed text-white font-extrabold text-[15px] rounded-2xl transition-all shadow-[0_0_30px_rgba(16,185,129,0.5)] hover:shadow-[0_0_40px_rgba(16,185,129,0.7)] uppercase tracking-widest flex items-center justify-center gap-2"
              ><Play className="w-5 h-5" /> Start Execution</button>
              {executionIssues.length > 0 && (
                <div className="rounded-2xl border border-amber-400/20 bg-amber-400/10 px-4 py-3 text-[13px] text-amber-200 leading-relaxed">
                  {executionIssues[0]}
                  {executionIssues.length > 1 ? `，另外還有 ${executionIssues.length - 1} 個 agent 尚未準備好。` : "。"}
                </div>
              )}
            </div>
          )}

          {guidelineContent && (
            <div className="space-y-3">
              <h3 className="text-[12px] font-black text-white/40 uppercase tracking-widest pl-1">Guideline Preview</h3>
              <div className="bg-black/20 border border-white/10 rounded-[20px] p-4 max-h-[300px] overflow-y-auto scrollbar-hide">
                <div data-color-mode="dark" className="markdown-transparent-bg">
                  <MarkdownPreview source={guidelineContent} style={{ backgroundColor: 'transparent', color: 'inherit', fontSize: '13px' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
