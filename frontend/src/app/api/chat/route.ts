import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

export const maxDuration = 300;

export async function POST(req: Request) {
  try {
    const { messages, model, mode, apiKey, task, payload } = await req.json();
    console.log("1. [Backend] Received Request - Model:", model, "Mode:", mode);

    const isOllama = (model ?? "").startsWith("ollama/");
    const isGemini = (model ?? "").startsWith("gemini");
    const actualModelName = isOllama ? model.replace("ollama/", "") : (model ?? "gpt-4o");
    const geminiModel = actualModelName.replace("-latest", "");

    let modelProvider;
    
    if (isGemini) {
      console.log("2. [Backend] Initializing Gemini Provider with model:", geminiModel);
      const google = createGoogleGenerativeAI({
        apiKey: apiKey || process.env.GOOGLE_GENERATIVE_AI_API_KEY || "invalid-key",
      });
      modelProvider = google(geminiModel);
    } else {
      console.log("2. [Backend] Initializing OpenAI/Ollama Provider with model:", actualModelName);
      const openai = createOpenAI({
        baseURL: isOllama ? "http://127.0.0.1:11434/v1" : "https://api.openai.com/v1",
        apiKey: apiKey || (isOllama ? "ollama" : "invalid-key"),
        fetch: async (url, options) => {
          console.log(`[Backend-Fetch] Requesting: ${url}`);
          // Remove any timeout signal from Next.js caching or Vercel boundaries
          const customOptions = { ...options };
          if (customOptions.signal) {
             delete customOptions.signal;
          }
          return fetch(url, customOptions);
        }
      });
      modelProvider = openai(actualModelName);
    }

    let systemPrompt = mode === "workspace" 
      ? "You are the workspace PM. Ask only the minimum clarifying questions needed, keep replies short and natural, and once requirements are clear produce a concise but actionable implementation guideline."
      : [
          "You are a normal end-user conversational assistant.",
          "Reply naturally, directly, and helpfully like a polished consumer chat app.",
          "Never pretend to be a PM, reviewer, planner, architect, or agent team unless the user explicitly asks.",
          "Do not mention implementation guideline, system report, workflow stage, tools, routing, constraints, or internal roles unless the user explicitly asks about them.",
          "If the user greets you, greet them back simply and naturally.",
        ].join(" ");

    if (task === "generate_agent_prompt") {
      const role = payload?.role || "Agent";
      const draftPrompt = payload?.draftPrompt || "";
      const guideline = payload?.guideline || "";
      systemPrompt =
        "You create concise system prompts for specialized AI agents. Reply only with the final system prompt in Traditional Chinese. The prompt should define role, responsibilities, allowed and forbidden actions, guideline compliance, response style, and escalation rules. For QA/reviewer roles, require item-by-item checking and explicit defect reporting.";
      const taskMessages = [
        {
          role: "user",
          content:
            `角色名稱：${role}\n\n` +
            `現有草稿：\n${draftPrompt || "（目前沒有草稿）"}\n\n` +
            `專案 guideline：\n${guideline || "（目前沒有 guideline）"}\n\n` +
            `請產生一份完整、可直接使用的 system prompt。`,
        },
      ];

      const result = await streamText({
        model: modelProvider,
        system: systemPrompt,
        messages: taskMessages,
        temperature: 0.4,
      });

      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const part of result.fullStream) {
              if (part.type === "text-delta") controller.enqueue(encoder.encode(part.text));
              else if (part.type === "error") throw part.error;
            }
          } finally {
            controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache" },
      });
    }

    if (task === "refine_guideline") {
      const guideline = payload?.guideline || "";
      const instruction = payload?.instruction || "";
      systemPrompt =
        "You revise implementation guidelines. Reply in Traditional Chinese only. Return only the updated guideline in clean Markdown. Preserve structure, improve clarity, and avoid excessive blank lines.";
      const taskMessages = [
        {
          role: "user",
          content:
            `目前 guideline：\n${guideline || "（空）"}\n\n` +
            `修改要求：\n${instruction || "請整理內容"}`,
        },
      ];

      const result = await streamText({
        model: modelProvider,
        system: systemPrompt,
        messages: taskMessages,
        temperature: 0.4,
      });

      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const part of result.fullStream) {
              if (part.type === "text-delta") controller.enqueue(encoder.encode(part.text));
              else if (part.type === "error") throw part.error;
            }
          } finally {
            controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache" },
      });
    }

    systemPrompt += "\n\n【最高指導原則 / CRITICAL INSTRUCTIONS】\n你必須使用「繁體中文 (Traditional Chinese, zh-TW)」進行所有的對話與回覆。絕對禁止使用任何簡體中文字。\n";

    console.log("3. [Backend] Calling streamText...");
    
    const result = await streamText({
      model: modelProvider,
      system: systemPrompt,
      messages,
      temperature: 0.7,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const part of result.fullStream) {
            if (part.type === 'text-delta') {
              controller.enqueue(encoder.encode(part.text));
            } else if (part.type === 'error') {
              throw part.error;
            }
          }
        } catch (streamErr: unknown) {
          const errorMessage = streamErr instanceof Error ? streamErr.message : 'Unknown error';
          console.error("❌ [Backend] Async Stream Error:", streamErr);
          controller.enqueue(encoder.encode(
            `\n⚠️ [API Error]: ${errorMessage}\n\nIf using Gemini, please check your API key.`
          ));
        } finally {
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-cache' }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error("❌ [Backend] Agent Streaming Error:", error);
    return new Response(JSON.stringify({ error: errorMessage }), { status: 500 });
  }
}
