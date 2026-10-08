from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


@dataclass
class ContactInfo:
    "Contact Information for a person"

    name: str
    email: str
    phone: str


agent = create_agent(model, response_format=ContactInfo)


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: John Doe, john@gmail.com, 7389620723",
            }
        ]
    }
)

print(response["structured_response"])
