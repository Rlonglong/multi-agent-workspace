import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

export const maxDuration = 60;

export async function POST(req: Request) {
  try {
    const { messages, model, mode, apiKey } = await req.json();
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
      });
      modelProvider = openai(actualModelName);
    }

    const systemPrompt = mode === "workspace" 
      ? "You are the Architect PM Agent. The user will describe an application idea. Actively gather requirements, structure out a complete frontend/backend component and module plan without coding directly. End with an 'Implementation Guideline' overview that the engineering agents can follow."
      : "You are a helpful and intelligent conversational AI assistant. Respond comprehensively but concisely to the user in a friendly tone.";

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
        } catch (streamErr: any) {
          console.error("❌ [Backend] Async Stream Error:", streamErr);
          controller.enqueue(encoder.encode(
            `\n⚠️ [API Error]: ${streamErr?.message ?? 'Unknown error'}\n\nIf using Gemini, please check your API key.`
          ));
        } finally {
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-cache' }
    });
  } catch (error: any) {
    console.error("❌ [Backend] Agent Streaming Error:", error);
    return new Response(JSON.stringify({ error: error?.message || String(error) }), { status: 500 });
  }
}
