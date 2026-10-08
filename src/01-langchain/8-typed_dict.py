from dotenv import load_dotenv
import os
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


# class MovieDict(TypedDict):
#     """A movie with details"""

#     title: Annotated[str, ..., "The title of the movie"]
#     year: Annotated[int, ..., "Release year of the movie"]
#     director: Annotated[str, ..., "Director of the movie"]
#     rating: Annotated[float, ..., "Rating of the movie"]

# model_with_typeddict=model.with_structured_output(MovieDict)
# response=model_with_typeddict.invoke("Provide details about the movie Interstellar")
# print(response)


class Actor(TypedDict):
    name: Annotated[str, ..., "The name of the actor"]
    role: Annotated[str, ..., "The role of the actor"]


class MovieDetails(TypedDict):
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "Release year of the movie"]
    cast: Annotated[list[Actor], ..., "The cast of the movie"]
    genres: Annotated[list[str], ..., "The genres of the movie"]
    budget: Annotated[float, ..., "The budget of the movie in million USD"]


model_with_typeddict = model.with_structured_output(MovieDetails)
response = model_with_typeddict.invoke("Provide details about the movie Interstellar")
print(response)
print(model.profile)
