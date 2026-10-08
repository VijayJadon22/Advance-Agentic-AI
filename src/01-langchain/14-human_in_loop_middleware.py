# Human In the Loop MiddleWare
# Pause agent execution for human approval, editing, or rejection of tool calls before they execute. Human-in-the-loop is useful for the following:

# High-stakes operations requiring human approval (e.g. database writes, financial transactions).
# Compliance workflows where human oversight is mandatory.
# Long-running conversations where human feedback guides the agent.

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.types import Command

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


def read_email_tool(email_id: str) -> str:
    """Mock function to read email by its id"""
    return f"Email content for ID: {email_id}"


def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject {subject}"


agent = create_agent(
    model,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        )
    ],
)

config = {"configurable": {"thread_id": "test"}}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Send email to john@gmail.com with subject what is the delivery update and body How are you? what is the update of the delivery of the parcel"
            )
        ]
    },
    config=config,
)

if "__interrupt__" in result:
    print("\n\n Paused Approving")

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=config
    )

print(f"Result: {result['messages'][-1].content}")
