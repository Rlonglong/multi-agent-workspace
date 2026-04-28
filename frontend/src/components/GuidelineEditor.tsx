"use client";

import { useState } from "react";
import { Bot, Play } from "lucide-react";

interface GuidelineEditorProps {
  initialContent: string;
  onStartWorkflow: (content: string) => void;
}

export function GuidelineEditor({ initialContent, onStartWorkflow }: GuidelineEditorProps) {
  const [content, setContent] = useState(initialContent);

  return (
    <div className="flex flex-col h-full rounded-[20px] overflow-hidden border border-white/10 shadow-2xl bg-white/5 backdrop-blur-md">
      <div className="flex items-center px-5 py-4 border-b border-white/10 bg-black/20">
        <Bot className="w-5 h-5 text-[#0a84ff] mr-3" />
        <div>
          <h2 className="text-sm font-semibold text-white/90">Guideline Editor</h2>
          <p className="text-[11px] text-white/50 tracking-wide">Plain text / Markdown draft. Modify specs before execution.</p>
        </div>
      </div>

      <div className="grid flex-1 min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
        <div className="min-h-0 border-b border-white/10 lg:border-b-0 lg:border-r lg:border-white/10">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="h-full min-h-[360px] w-full resize-none bg-transparent px-5 py-4 text-[14px] leading-7 text-white/90 outline-none placeholder:text-white/25"
            placeholder="在這裡調整 implementation guideline..."
          />
        </div>
        <div className="min-h-0 bg-black/15">
          <div className="border-b border-white/10 px-5 py-3 text-[11px] uppercase tracking-[0.22em] text-white/35">
            Preview
          </div>
          <div className="h-full overflow-y-auto px-5 py-4">
            <pre className="whitespace-pre-wrap break-words text-[13px] leading-7 text-white/75">
              {content || "尚未輸入內容"}
            </pre>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-white/10 bg-black/30 flex justify-end">
        <button 
          onClick={() => onStartWorkflow(content)} 
          className="bg-[#0a84ff] hover:bg-[#007aff] text-white font-medium text-sm shadow-[0_0_20px_rgba(10,132,255,0.3)] px-6 py-2.5 rounded-full transition-all hover:scale-[1.02] flex items-center"
        >
          <Play className="w-3.5 h-3.5 mr-2 fill-current" />
          Approve & Start Agents
        </button>
      </div>
    </div>
  );
}
