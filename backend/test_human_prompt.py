import asyncio, httpx, json

async def stream():
    # Simulate old breakage (Huge System Message) -> This breaks <think>
    # Simulate proposed fix (Huge User Message)
    payload = {
        "model": "deepseek-r1:32b",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Here are your instructions: You are a PM Architect. Output a highly detailed guideline... Now, my prompt is: 你好嗎"}
        ],
        "stream": True,
        "options": {"temperature": 0.6}
    }
    has_thinking = False
    print("\n--- Testing HumanMessage structure ---")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:11434/api/chat", json=payload, timeout=30.0) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    if msg.get("thinking") or "<think>" in msg.get("content", ""):
                        has_thinking = True
                        break
            print(f"Has <think> or thinking field (User Mode): {has_thinking}")

asyncio.run(stream())
