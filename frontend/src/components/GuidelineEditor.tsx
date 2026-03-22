"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { Bot, Play } from "lucide-react";
import "@uiw/react-md-editor/markdown-editor.css";
import "@uiw/react-markdown-preview/markdown.css";

const MDEditor = dynamic(
  () => import("@uiw/react-md-editor").then((mod) => mod.default),
  { ssr: false }
);

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
          <p className="text-[11px] text-white/50 tracking-wide">Markdown supported. Modify specs before execution.</p>
        </div>
      </div>
      
      <div className="flex-1 w-full relative" data-color-mode="dark">
        <MDEditor
          value={content}
          onChange={(val) => setContent(val || "")}
          height="100%"
          className="w-full h-full border-none rounded-none !bg-transparent text-[14px]"
          preview="live"
        />
        <style>{`
          .wmde-markdown { background-color: transparent !important; color: #f5f5f7 !important; }
          .w-md-editor-toolbar { background-color: rgba(255,255,255,0.05) !important; border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
          .w-md-editor-content { background-color: transparent !important; }
        `}</style>
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
