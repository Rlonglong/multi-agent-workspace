"use client";

import { useState, useEffect, useRef } from "react";
import { AgentConfig } from "@/components/ModelSelector";
import { GuidelineEditor } from "@/components/GuidelineEditor";
import { Send, MessageSquare, LayoutPanelLeft, Sparkles, ChevronDown, Clock, KeyRound, Bot, Rocket, Info, X, PanelLeftClose, PanelLeftOpen } from "lucide-react";

export interface ChatMessage {
  id: string;
  role: "system" | "user" | "assistant" | "data";
  content: string;
}

export type ProjectMode = "chat" | "workspace";

export interface Session {
  id: string;
  title: string;
  mode: ProjectMode;
  model: string;
  messages: ChatMessage[];
  guideline: string | null;
  updatedAt: number;
}

const DEFAULT_AGENTS: AgentConfig[] = [
  { role: "PM", model: "gpt-4o", prompt: "You are the Project Manager. Output the requirements and guideline." },
  { role: "Frontend", model: "claude-3-5-sonnet-20240620", prompt: "You are the Frontend Architect." },
  { role: "Backend", model: "gemini-2.5-pro", prompt: "You are the Backend Architect." },
  { role: "QA", model: "ollama/llama3.2", prompt: "You are the QA Tester." },
  { role: "Marketing", model: "ollama/qwen2.5", prompt: "You are the Marketing Expert." },
];

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [agentConfigs, setAgentConfigs] = useState<AgentConfig[]>(DEFAULT_AGENTS);

  const [localInput, setLocalInput] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Custom model UI states
  const [isCustomModel, setIsCustomModel] = useState(false);
  const [customModelId, setCustomModelId] = useState("");
  const [showHelp, setShowHelp] = useState(false);

  // Spotlight Glassmorphism
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isMouseOver, setIsMouseOver] = useState(false);
  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem("multi-agent-sessions");
    const storedKey = localStorage.getItem("multi-agent-apikey");
    if (storedKey) setApiKey(storedKey);

    if (stored) {
      const parsed = JSON.parse(stored) as Session[];
      setSessions(parsed);
      if (parsed.length > 0) {
        setCurrentSessionId(parsed[0].id);
      } else {
        createNewSession("chat");
      }
    } else {
      createNewSession("chat");
    }
  }, []);

  useEffect(() => {
    if (mounted) localStorage.setItem("multi-agent-apikey", apiKey);
  }, [apiKey, mounted]);

  const createNewSession = (mode: ProjectMode) => {
    const newSession: Session = {
      id: Date.now().toString(),
      title: "New " + (mode === "workspace" ? "Project" : "Chat"),
      mode,
      model: "gpt-4o",
      messages: [],
      guideline: null,
      updatedAt: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
  };

  const currentSession = sessions.find((s) => s.id === currentSessionId);

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
      localStorage.setItem("multi-agent-sessions", JSON.stringify(next));
      return next;
    });
  };

  const submitChat = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!localInput.trim() || isLoading || !currentSessionId) return;

    const userMsg: ChatMessage = { id: Date.now().toString(), role: "user", content: localInput };
    const newMessages = [...(currentSession?.messages || []), userMsg];

    updateSession(currentSessionId, { messages: newMessages });
    setLocalInput("");
    setIsLoading(true);

    const assistantId = (Date.now() + 1).toString();
    updateSession(currentSessionId, {
      messages: [...newMessages, { id: assistantId, role: "assistant", content: "" }]
    });

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages,
          mode: currentSession?.mode || "chat",
          model: currentSession?.mode === "workspace" ? agentConfigs.find(a => a.role === "PM")?.model : currentSession?.model,
          apiKey: apiKey
        })
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
    } catch (err: any) {
      console.error(err);
      updateSession(currentSessionId, {
        messages: [...newMessages, { id: assistantId, role: "assistant", content: `⚠️ [Network Error]: Could not connect to the chat server.\nDetail: ${err.message}` }]
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfigChange = (newConfig: AgentConfig) => {
    setAgentConfigs((prev) => prev.map((c) => (c.role === newConfig.role ? newConfig : c)));
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentSession?.messages]);

  useEffect(() => {
    document.title = "Rlong GPT";
  }, []);

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
            <button
              key={s.id}
              onClick={() => { setCurrentSessionId(s.id); setIsPanelOpen(false); }}
              className={`w-full text-left truncate px-4 py-3 rounded-[20px] text-[14px] font-medium transition-all ${currentSessionId === s.id ? "bg-white/20 text-white shadow-md border border-white/10" : "text-white/50 hover:bg-white/10 hover:text-white/90"}`}
            >
              <span className="opacity-70 mr-1">{s.mode === "workspace" ? "🚀" : "💬"}</span> {s.title}
            </button>
          ))}
        </div>
      </div>

      {/* Floating Curved Main Chat Area */}
      <div className={`flex-1 flex flex-col relative transition-all duration-500 ease-out z-10 bg-white/[0.01] backdrop-blur-[6px] border-[1.5px] border-white/20 rounded-[32px] overflow-hidden shadow-[inset_0_0_20px_rgba(255,255,255,0.03),0_10px_40px_rgba(0,0,0,0.6)] ${isPanelOpen ? "xl:mr-[340px] 2xl:mr-[420px]" : ""}`}>

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

              {!(currentSession.model?.startsWith("ollama/")) && (
                <div className="hidden sm:flex items-center bg-white/5 px-3 py-2 rounded-[18px] border border-white/10 text-sm overflow-hidden focus-within:border-indigo-500 focus-within:bg-black/40 focus-within:shadow-[0_0_20px_rgba(99,102,241,0.2)] transition-all">
                  <KeyRound className="w-4 h-4 text-white/50 mx-1" />
                  <input
                    type="password"
                    placeholder="API Key..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="bg-transparent text-[13px] w-[110px] xl:w-[150px] outline-none text-white placeholder:text-white/30 font-medium"
                  />
                </div>
              )}
              <div className="flex items-center gap-1 bg-white/5 hover:bg-white/10 px-4 py-2 rounded-[18px] border border-white/10 text-sm shadow-sm transition-all focus-within:border-indigo-500 focus-within:bg-black/40 focus-within:shadow-[0_0_20px_rgba(99,102,241,0.2)]">
                <select
                  value={isCustomModel ? "custom" : (currentSession.mode === "workspace" ? agentConfigs.find(a => a.role === "PM")?.model || "gpt-4o" : currentSession.model)}
                  onChange={(e) => {
                    const newVal = e.target.value;
                    if (newVal === "custom") {
                      setIsCustomModel(true);
                      updateSession(currentSessionId, { model: customModelId || "ollama/mistral" });
                    } else {
                      setIsCustomModel(false);
                      updateSession(currentSessionId, { model: newVal });
                      if (currentSession.mode === "workspace") {
                        setAgentConfigs(configs => configs.map(c => c.role === "PM" ? { ...c, model: newVal } : c));
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
                      setCustomModelId(e.target.value);
                      updateSession(currentSessionId, { model: e.target.value });
                      if (currentSession.mode === "workspace") {
                        setAgentConfigs(configs => configs.map(c => c.role === "PM" ? { ...c, model: e.target.value } : c));
                      }
                    }}
                    className="bg-black/50 border border-indigo-500 text-white text-[13px] px-3 py-1.5 rounded-xl outline-none w-[140px] ml-3 shadow-[0_0_15px_rgba(99,102,241,0.2)] font-mono"
                  />
                )}
              </div>
            </div>

            {currentSession.mode === "workspace" && (
              <button
                onClick={() => setIsPanelOpen(!isPanelOpen)}
                className="rounded-[18px] bg-white/10 border border-white/10 hover:bg-white/20 transition-all p-2.5 text-white shadow-md hover:scale-105 ml-2"
              >
                <LayoutPanelLeft className="w-5 h-5" />
              </button>
            )}
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto w-full px-5 sm:px-8 scroll-smooth relative z-20 scrollbar-hide">
          <div className="max-w-4xl mx-auto py-10 space-y-8 min-h-full pb-8">
            {currentSession.messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center mt-[15vh] space-y-6 opacity-95">
                {currentSession.mode === "workspace" ? (
                  <><div className="w-[96px] h-[96px] bg-black/20 border border-white/10 backdrop-blur-2xl rounded-[32px] shadow-[0_0_50px_rgba(99,102,241,0.4)] flex items-center justify-center p-3">
                    <img src="/logo.svg" className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]" />
                  </div>
                    <h2 className="text-[32px] font-extrabold tracking-tight text-white drop-shadow-lg">Architect PM Agent</h2>
                    <p className="max-w-lg text-white/50 text-[16px] font-medium leading-relaxed">Describe your application vision. I will analyze the requirements and draft an implementation guideline for the engineering agents.</p></>
                ) : (
                  <><div className="w-[96px] h-[96px] bg-black/20 border border-white/10 backdrop-blur-2xl rounded-[32px] shadow-[0_0_50px_rgba(10,132,255,0.4)] flex items-center justify-center p-3">
                    <img src="/logo.svg" className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]" />
                  </div>
                    <h2 className="text-[32px] font-extrabold tracking-tight text-white drop-shadow-lg">How can I help you today?</h2></>
                )}
              </div>
            )}

            {currentSession.messages.map((m) => (
              <div key={m.id} className={`flex gap-5 group ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {m.role === 'assistant' && (
                  <div className="w-[38px] h-[38px] rounded-[14px] bg-black/30 backdrop-blur-xl shrink-0 flex items-center justify-center shadow-[0_5px_15px_rgba(0,0,0,0.5)] border border-white/10 mt-1 p-1.5">
                    <img src="/logo.svg" alt="Bot" className="w-full h-full object-contain drop-shadow-md" />
                  </div>
                )}

                <div className={`max-w-[85%] prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10 break-words ${m.role === 'user' ? 'bg-[#0a84ff] text-white px-6 py-4 rounded-[28px] rounded-br-[8px] shadow-[0_10px_30px_rgba(10,132,255,0.25)] backdrop-blur-md border border-[#32ade6]/30 w-fit ml-auto' : 'flex-1 bg-white/[0.01] backdrop-blur-[6px] border border-white/15 px-6 py-4 rounded-[28px] rounded-bl-[8px] shadow-[0_5px_20px_rgba(0,0,0,0.2)] transition-all duration-500'}`}>
                  {m.role === 'assistant' && (
                    <div className="font-bold text-[11px] mb-2.5 text-white/40 uppercase tracking-widest flex items-center gap-1.5">
                      {currentSession.mode === 'workspace' ? <Rocket className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                      {currentSession.mode === 'workspace' ? 'PM Agent' : 'Rlong GPT'}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap text-[15.5px] font-medium leading-[1.65] drop-shadow-sm">{m.content}</div>
                </div>
              </div>
            ))}
            {isLoading && currentSession.messages[currentSession.messages.length - 1]?.role === "user" && (
              <div className="flex gap-5">
                <div className="w-[38px] h-[38px] rounded-[14px] bg-black/30 backdrop-blur-xl shrink-0 flex items-center justify-center shadow-lg border border-white/10 mt-1 p-1.5 animate-pulse">
                  <img src="/logo.svg" alt="Bot" className="w-full h-full object-contain" />
                </div>
                <div className="bg-white/[0.01] backdrop-blur-[6px] border border-white/15 px-6 py-5 rounded-[28px] rounded-bl-[8px] flex items-center gap-2 transition-all duration-500">
                  <div className="w-2.5 h-2.5 bg-white/40 rounded-full animate-bounce"></div>
                  <div className="w-2.5 h-2.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                  <div className="w-2.5 h-2.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="w-full shrink-0 bg-transparent pt-2 pb-6 px-6 z-30">
          <div className="max-w-4xl mx-auto relative group">
            <form onSubmit={submitChat}>
              <textarea
                className="w-full resize-none bg-white/[0.02] backdrop-blur-[8px] border-[1.5px] border-white/25 hover:border-white/40 rounded-[32px] pl-7 pr-16 py-5 text-[15.5px] placeholder-white/40 scrollbar-hide focus-animated outline-none shadow-[inset_0_0_20px_rgba(255,255,255,0.03),0_20px_50px_rgba(0,0,0,0.6)] font-medium text-white transition-all duration-500 ease-out"
                placeholder={currentSession.mode === "workspace" ? "Describe your project idea..." : "Message AI..."}
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
                type="submit"
                disabled={isLoading || !localInput.trim()}
                className={`absolute right-4 bottom-4 p-3 rounded-[20px] transition-all duration-300 ${localInput.trim() ? (currentSession.mode === 'workspace' ? 'bg-indigo-500 text-white shadow-[0_0_25px_rgba(99,102,241,0.6)] hover:scale-110' : 'bg-white text-black shadow-[0_0_25px_rgba(255,255,255,0.6)] hover:scale-110') : 'bg-white/10 text-white/30'}`}
              >
                <Send className="w-[20px] h-[20px]" />
              </button>
            </form>
          </div>
          <div className="text-center mt-5 text-[11px] text-white/30 tracking-widest font-extrabold uppercase drop-shadow-md">
            Segora © 2026 All Right Reserved.
          </div>
        </div>
      </div>

      {/* Right Artifact Panel */}
      <div
        className={`fixed top-4 bottom-4 right-4 bg-white/[0.01] backdrop-blur-[6px] border-[1.5px] border-white/20 rounded-[32px] z-[40] flex flex-col shadow-[inset_0_0_20px_rgba(255,255,255,0.03),-20px_0_50px_rgba(0,0,0,0.6)] transition-all duration-500 ease-out overflow-hidden ${isPanelOpen ? "w-[320px] xl:w-[400px] translate-x-0 opacity-100" : "w-0 translate-x-full opacity-0 border-none shadow-none"}`}
      >
        <div className="flex items-center justify-between p-6 border-b border-white/10 bg-black/10">
          <h2 className="font-exrabold text-[17px] tracking-tight text-white">Project Configuration</h2>
          <button onClick={() => setIsPanelOpen(false)} className="text-white/50 hover:text-white p-2 flex items-center justify-center rounded-full hover:bg-white/10 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8 scrollbar-hide">
          <div className="space-y-4">
            <h3 className="text-[12px] font-black text-white/40 uppercase tracking-widest pl-1">Agent Assignment & Context</h3>
            <div className="bg-black/20 border border-white/10 rounded-[24px] p-4 space-y-4 shadow-inner backdrop-blur-xl">
              {agentConfigs.map((config) => (
                <div key={config.role} className="flex flex-col gap-2 py-3 border-b border-white/5 last:border-0 hover:bg-white/5 rounded-2xl transition-colors px-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <div className="w-7 h-7 rounded-lg bg-white/10 flex items-center justify-center border border-white/5 p-1.5 shadow-sm">
                        <img src="/logo.svg" className="w-full h-full opacity-90" />
                      </div>
                      <span className="font-bold text-[14px] text-white/90">{config.role}</span>
                    </div>
                    <select
                      value={config.model}
                      onChange={(e) => handleConfigChange({ ...config, model: e.target.value })}
                      className="bg-black/50 border border-white/10 text-white text-[12px] px-2 py-1.5 rounded-lg outline-none max-w-[140px]"
                    >
                      <option value="gpt-4o" className="text-black">GPT-4o</option>
                      <option value="claude-3-5-sonnet-20240620" className="text-black">Claude 3.5</option>
                      <option value="gemini-2.5-flash" className="text-black">Gemini 2.5 Flash</option>
                      <option value="gemini-2.5-pro" className="text-black">Gemini 2.5 Pro</option>
                      <option disabled className="text-black">──────────</option>
                      <option value="ollama/deepseek-r1:32b" className="text-black">DeepSeek R1 32B (Local)</option>
                      <option value="ollama/llama3.2" className="text-black">Llama 3.2 (Local)</option>
                      <option value="ollama/qwen2.5" className="text-black">Qwen 2.5 (Local)</option>
                    </select>
                  </div>
                  <textarea
                    placeholder={`${config.role} custom prompt instructions...`}
                    value={config.prompt || ""}
                    onChange={(e) => handleConfigChange({ ...config, prompt: e.target.value })}
                    className="w-full h-[60px] mt-1 bg-black/40 border border-white/10 hover:border-white/20 rounded-[14px] px-3 py-2 text-[12.5px] text-white/80 placeholder-white/30 resize-none outline-none focus:border-indigo-400/50 transition-colors"
                  />
                </div>
              ))}
            </div>
          </div>

          {currentSession.guideline && (
            <div className="space-y-4 flex-1 flex flex-col h-[550px]">
              <h3 className="text-[12px] font-black text-white/40 uppercase tracking-widest pl-1">Architect Guideline</h3>
              <div className="flex-1 min-h-[400px]">
                <GuidelineEditor initialContent={currentSession.guideline} onStartWorkflow={(c) => alert("Workflow approved! " + c.slice(0, 10))} />
              </div>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
