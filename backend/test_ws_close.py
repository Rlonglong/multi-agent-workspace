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
        
        has_think_close = False
        print("Connected. Listening for chunks...")
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                if data["type"] == "token":
                    content = data.get("content", "")
                    if "</think>" in content:
                        has_think_close = True
                        print("\n>>> DETECTED </think>! <<<")
                        break
                    print(content, end="", flush=True)
                elif data["type"] in ["error", "finish"]:
                    print("\nStream finished.")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print(f"\nFinal: did backend send </think> tag? {has_think_close}")

asyncio.run(test())
