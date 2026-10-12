from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": r"D:\PythonProjects\agentic-ai\.venv\Scripts\python.exe",
                "args": [
                    r"D:\PythonProjects\agentic-ai\src\04-mcp-practice\math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()
    model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")

    agent = create_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is (3+5) x 12?"}]}
    )
    print(math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "Hi what is the weather like in gwalior?"}
            ]
        }
    )
    print(weather_response["messages"][-1].content)


asyncio.run(main())
