from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")
tool = TavilySearch(max_results=2)
memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def superbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("superbot", superbot)

graph_builder.add_edge(START, "superbot")
graph_builder.add_edge("superbot", END)

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user1"}}

# Invoke way
# response = graph.invoke(
#     {
#         "messages": "Hi my name is Vijay Jadon and i love football my favorite team is Real Madrid and favorite player is Cristiano Ronaldo."
#     },
#     config=config,
# )

# print(response["messages"][-1].content)


# ----------------------***---------------------
# Streaming
# Methods: .stream() and astream()

# These methods are sync and async methods for streaming back results.
# Additional parameters in streaming modes for graph state

# values : This streams the full state of the graph after each node is called.
# updates : This streams updates to the state of the graph after each node is called.

for chunk in graph.stream(
    {
        "messages": "Hi my name is Vijay Jadon and I love football my favorite team is Real Madrid and favorite player is Cristiano Ronaldo."
    },
    config=config,
    stream_mode="updates",
):
    print(chunk)

