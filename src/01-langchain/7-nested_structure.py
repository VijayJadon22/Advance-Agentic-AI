from dotenv import load_dotenv
import os

load_dotenv()
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


class Actor(BaseModel):
    name: str
    role: str


class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")


model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Ocean's Eleven")
print(response)
