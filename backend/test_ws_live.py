import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        payload = {
            "messages": [{"id": "1", "role": "user", "content": "你好嗎"}],
            "agent_configs": [{"role": "PM", "model": "ollama/deepseek-r1:32b", "prompt": "You are the central Architect PM."}],
            "mode": "workspace",
            "api_key": "",
            "stage": "discovery",
            "guideline": ""
        }
        await websocket.send(json.dumps(payload))
        
        has_thinking_tag = False
        print("Connected. Listening for chunks...")
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "token":
                print(repr(data["content"]))
                if "<think>" in data["content"]:
                    has_thinking_tag = True
                    break
            elif data["type"] in ["error", "finish"]:
                break
        
        print(f"\nFinal: did backend send <think> tag? {has_thinking_tag}")

asyncio.run(test())
