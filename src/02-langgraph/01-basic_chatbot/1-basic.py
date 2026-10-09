# Build a basic chatbot with LangGrpah(Graph API)

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# Node functionality
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# response = graph.invoke({"messages": "Hi There"})
# print(response["messages"][0].content)
# print(response["messages"][1].content)

for event in graph.stream({"messages": "Hi There How are you"}):
    for value in event.values():
        print(value["messages"][-1].content)
