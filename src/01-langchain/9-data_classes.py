from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing_extensions import TypedDict

load_dotenv()

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


# class ContactInfo(BaseModel):
#     """Contact information for a person."""

#     name: str = Field(..., description="Name of the person")
#     email: str = Field(..., description="Email of the person")
#     phone: str = Field(..., description="Contact number of the person")


# agent = create_agent(
#     model, response_format=ContactInfo
# )  # autoselects provider stargety

# response = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Extract contact info from: John Doe, john@gmail.com, 7389620723",
#             }
#         ]
#     }
# )

# print(response["structured_response"])


class ContactInfo(TypedDict):
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
