from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai")

tool = TavilySearch(max_results=2)
memory = MemorySaver()


def multiply(a: int, b: int) -> int:
    """Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a * b


tools = [tool, multiply]

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("tool_calling_llm", tool_calling_llm)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "tool_calling_llm")
graph_builder.add_conditional_edges("tool_calling_llm", tools_condition)
graph_builder.add_edge("tools", "tool_calling_llm")

graph = graph_builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "user1"}}

response = graph.invoke({"messages": "Hi my name is Vijay Jadon"}, config=config)
print(response["messages"][-1].content)

result = graph.invoke({"messages": "Hi what is my name?"}, config=config)
print(result["messages"][-1].content)
