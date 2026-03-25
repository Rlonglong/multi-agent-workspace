import asyncio, httpx, json
from app.agents.llm_router import LocalOllamaChatModel
from langchain_core.messages import SystemMessage, HumanMessage

async def run():
    llm = LocalOllamaChatModel(model_name="deepseek-r1:32b", temperature=0.7)
    messages = [
        SystemMessage(content="You are a smart assistant. You must think step by step before answering. Answer in Traditional Chinese."), 
        HumanMessage(content="你好")
    ]
    
    print("Starting generator...")
    final = ""
    async for chunk in llm._astream(messages):
        final += chunk.message.content
        print(repr(chunk.message.content))
    print("\nFINAL TEXT:")
    print(final)

asyncio.run(run())
