from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio

from langchain_groq import ChatGroq
# from langchain.chat_models import init_chat_model
import os

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": r"D:\PythonProjects\agentic-ai\.venv\Scripts\python.exe",
                "args": [r"D:\PythonProjects\agentic-ai\src\03-mcp\mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()
    model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    agent = create_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is (3+5) x 12?"}]}
    )

    print(math_response["messages"][-1].content)


asyncio.run(main())
