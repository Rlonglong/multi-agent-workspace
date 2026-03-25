import asyncio
import json
import httpx

async def stream():
    heavy_instructions = (
        "You are the central Architect PM in a multi-agent AI software development team. "
        "\n\n【最高指導原則 / CRITICAL INSTRUCTIONS】\n"
        "1. ABSOLUTELY NEVER use '---' or horizontal rules anywhere in your output. NEVER.\n"
        "2. 絕對禁止使用「簡體中文 (Simplified Chinese)」。\n"
        "4. CRITICAL XML LOGIC: You MUST explicitly wrap your internal thought process inside `<think>` and `</think>` tags BEFORE answering the user! NEVER forget the `</think>` closing tag.\n"
        "---\nUser Request Below:\n\n"
    )
    payload = {
        "model": "deepseek-r1:32b",
        "messages": [
             {"role": "system", "content": "You are a highly capable reasoning AI assistant."},
             {"role": "user", "content": heavy_instructions + "我要做一個網站"}
        ],
        "stream": True,
        "options": {"temperature": 0.6}
    }
    
    print("\n[OLLAMA RAW CHUNK TEST]\n")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:11434/api/chat", json=payload, timeout=30.0) as response:
            count = 0
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    thinking = msg.get("thinking")
                    content = msg.get("content")
                    print(f"Chunk {count}: thinking={repr(thinking)} content={repr(content)}")
                    count += 1
                    if count > 8:
                        break

asyncio.run(stream())
