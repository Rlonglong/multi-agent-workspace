import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.graph import pm_prompt

async def run():
    llm = ChatOpenAI(
        model="deepseek-r1:32b", 
        base_url="http://127.0.0.1:11434/v1", 
        api_key="ollama",
        temperature=0.2,
        streaming=True
    )
    messages = [SystemMessage(content=pm_prompt), HumanMessage(content="你好")]
    print("Starting generator...")
    final = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            final += chunk.content
            print(repr(chunk.content))
    print("\nFINAL TEXT:")
    print(final)

asyncio.run(run())
