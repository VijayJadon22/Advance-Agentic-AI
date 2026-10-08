from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search_hotels(city: str) -> str:
    """Search hotels - return long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")

agent = create_agent(
    model=model,
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model, trigger=("tokens", 550), keep=("tokens", 200)
        )
    ],
)

config = {"configurable": {"thread_id": "vijay"}}


# Token Counter
def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4  # 4chars=1token


cities = ["Gwalior", "gurgaon", "Jaipur", "Delhi", "Indore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]}, config=config
    )

    tokens = count_tokens(response["messages"])
    print("\n" + "=" * 70)
    print(f"🏨 CITY       : {city}")
    print(f"📊 TOKENS     : ~{tokens}")
    print(f"💬 MESSAGES   : {len(response['messages'])}")
    print("=" * 70)

    for i, message in enumerate(response["messages"], 1):
        print(f"\n--- Message {i}: {message.__class__.__name__} ---")
        print(message.content)
