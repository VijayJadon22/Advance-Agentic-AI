from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")

agent = create_agent(
    model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 10),
            keep=("messages", 4),
        )
    ],
)

# Run with thread id
config = {"configurable": {"thread_id": "vijay_123"}}

questions = [
    "Hi my name is Vijay",
    "Who are you?",
    "I am from gwalior currently in guragon working",
    "I am working as a full stack developer here",
    "Its been more than 1 year i am in gurgaon",
    "Diwali is coming i will be going to my hometwon gwalior",
]

for q in questions:
    response = agent.invoke({"messages": [HumanMessage(content=q)]}, config=config)
    print(f"Messages: {response}")
    print(f"Messages: {len(response['messages'])}")
