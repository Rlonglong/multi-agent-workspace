import asyncio, httpx, json
from app.agents.graph import pm_prompt

async def stream_ollama(prompt):
    payload = {
        "model": "deepseek-r1:32b",
        "messages": [{"role": "system", "content": prompt},{"role": "user", "content": "你好嗎"}],
        "stream": True,
        "options": {"temperature": 0.2}
    }
    print(f"\n--- TESTING PROMPT LENGTH: {len(prompt)} ---")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:11434/api/chat", json=payload, timeout=30.0) as response:
            has_thinking = False
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
            print(f"Has <think> or thinking field: {has_thinking}")

async def run():
    await stream_ollama("You are a helpful AI assistant.")
    await stream_ollama(pm_prompt)

asyncio.run(run())
