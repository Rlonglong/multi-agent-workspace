import asyncio, httpx, json
from app.agents.graph import pm_prompt

async def stream_ollama():
    payload = {
        "model": "deepseek-r1:32b",
        "messages": [{"role": "system", "content": pm_prompt},{"role": "user", "content": "你好嗎"}],
        "stream": True,
        "options": {"temperature": 0.6}
    }
    has_thinking = False
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:11434/api/chat", json=payload, timeout=30.0) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    if msg.get("thinking"): 
                        has_thinking = True
                        break
                    elif "<think>" in msg.get("content", ""):
                        has_thinking = True
                        break
            print(f"Has <think> or thinking field at temp 0.6: {has_thinking}")

asyncio.run(stream_ollama())
