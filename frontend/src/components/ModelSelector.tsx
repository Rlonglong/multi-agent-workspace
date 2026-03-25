"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type AgentRole = string;

export interface AgentConfig {
  role: AgentRole;
  model: string;
  prompt?: string;
  apiKey?: string;
}

const DEFAULT_MODELS = [
  { value: "gpt-4o", label: "GPT-4o (OpenAI)" },
  { value: "gpt-4o-mini", label: "GPT-4o Mini" },
  { value: "claude-3-5-sonnet-20240620", label: "Claude 3.5 Sonnet" },
  { value: "ollama/llama3.2", label: "Llama 3.2 (Local Ollama)" },
  { value: "ollama/qwen2.5", label: "Qwen 2.5 (Local Ollama)" },
  { value: "custom", label: "Custom API string..." }
];

export function ModelSelector({
  config,
  onChange,
}: {
  config: AgentConfig;
  onChange: (newConfig: AgentConfig) => void;
}) {
  const [isCustom, setIsCustom] = useState(false);

  return (
    <div className="flex items-center space-x-4 mb-4">
      <div className="w-24 font-bold text-sm text-foreground/80 tracking-wide uppercase">{config.role}</div>
      <div className="flex-1 max-w-sm">
        <Select
          value={isCustom ? "custom" : config.model}
          onValueChange={(val) => {
            if (val === "custom") {
              setIsCustom(true);
            } else {
              setIsCustom(false);
              onChange({ ...config, model: val || "" });
            }
          }}
        >
          <SelectTrigger className="bg-background">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {DEFAULT_MODELS.map((m) => (
              <SelectItem key={m.value} value={m.value}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {isCustom && (
        <div className="flex-1 max-w-sm animate-in fade-in zoom-in duration-300">
          <Input 
            placeholder="e.g., ollama/mistral" 
            value={config.model} 
            onChange={(e) => onChange({ ...config, model: e.target.value })}
            className="bg-background"
          />
        </div>
      )}
    </div>
  );
}
