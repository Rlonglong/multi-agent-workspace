import asyncio
import os
import json
import httpx
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

OLLAMA_REQUEST_SEMAPHORE = asyncio.Semaphore(int(os.getenv("OLLAMA_MAX_CONCURRENT_REQUESTS", "1")))
OLLAMA_REQUEST_DELAY_SECONDS = float(os.getenv("OLLAMA_REQUEST_DELAY_SECONDS", "0.35"))

class LocalOllamaChatModel(BaseChatModel):
    model_name: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "local_ollama_custom"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Only async stream is supported for this custom wrapper.")

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        final_content = ""
        async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            if chunk.message.content:
                final_content += chunk.message.content
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=final_content))])

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        # Format messages for Ollama
        formatted_messages = []
        for m in messages:
            role = "user"
            if m.type == "ai": role = "assistant"
            elif m.type == "system": role = "system"
            formatted_messages.append({"role": role, "content": m.content})
            
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,
            "options": {"temperature": self.temperature}
        }
        
        in_thinking_block = False
        
        async with OLLAMA_REQUEST_SEMAPHORE:
            await asyncio.sleep(OLLAMA_REQUEST_DELAY_SECONDS)
            async with httpx.AsyncClient() as client:
                yielded_any = False
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=300.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            msg = data.get("message", {})
                            
                            # Ollama native API uses "thinking" field for DeepSeek-R1
                            thinking = msg.get("thinking")
                            content = msg.get("content")
                            
                            chunk_text = ""
                            if thinking:
                                if not in_thinking_block:
                                    chunk_text += "<think>\n"
                                    in_thinking_block = True
                                chunk_text += thinking
                                
                            if content:
                                if in_thinking_block:
                                    chunk_text += "\n</think>\n\n"
                                    in_thinking_block = False
                                chunk_text += content
                                
                            if chunk_text:
                                yielded_any = True
                                chunk = AIMessageChunk(content=chunk_text)
                                yield ChatGenerationChunk(message=chunk)
                        except json.JSONDecodeError:
                            continue

                if not yielded_any:
                    non_stream_payload = {**payload, "stream": False}
                    await asyncio.sleep(OLLAMA_REQUEST_DELAY_SECONDS)
                    response = await client.post(f"{self.base_url}/api/chat", json=non_stream_payload, timeout=300.0)
                    response.raise_for_status()
                    data = response.json()
                    msg = data.get("message", {})
                    thinking = msg.get("thinking")
                    content = msg.get("content")
                    chunk_text = ""
                    if thinking:
                        chunk_text += "<think>\n"
                        chunk_text += str(thinking)
                        if content:
                            chunk_text += "\n</think>\n\n"
                    if content:
                        chunk_text += str(content)
                    if chunk_text:
                        chunk = AIMessageChunk(content=chunk_text)
                        yield ChatGenerationChunk(message=chunk)

def get_llm(model_str: str, temperature: float = 0.7) -> BaseChatModel:
    """
    Returns a unified LangChain ChatModel interface.
    Uses our custom wrapper for Ollama to salvage DeepSeek `<think>` tags that LangChain drops.
    """
    api_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")

    if model_str.startswith("ollama/"):
        model_name = model_str.replace("ollama/", "")
        return LocalOllamaChatModel(
            model_name=model_name, 
            temperature=temperature, 
            base_url=api_base
        )
    elif model_str.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=model_str, 
            temperature=temperature
        )
    elif model_str.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_str, 
            temperature=temperature
        )
    else:
        # Default to OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_str, 
            temperature=temperature,
            streaming=True
        )
