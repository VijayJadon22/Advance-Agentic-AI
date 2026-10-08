from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


# class Movie(BaseModel):
#     title: str = Field(description="Title of the movie")
#     year: int = Field(description="Release year of the movie")
#     director: str = Field(description="Director of the movie")
#     rating: float = Field(description="Rating of the movie")

# model_with_structure = model.with_structured_output(Movie)

# response = model_with_structure.invoke(
#     "Provide details about the movie Shwashank Redemption"
# )
# print(response)


class Movie(BaseModel):
    title: str = Field(..., description="Title of the movie")
    year: int = Field(..., description="Release year of the movie")
    director: str = Field(..., description="Director of the movie")
    rating: float = Field(..., description="Rating of the movie")


model_with_structure = model.with_structured_output(Movie,include_raw=True)

response = model_with_structure.invoke(
    "Provide details about the movie Shwashank Redemption"
)
print(response)
