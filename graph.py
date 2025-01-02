from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
import streamlit as st
# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys

openai_api_key = st.secrets["openai_api_key"]
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)


# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)
    response = llm.invoke(messages)
    return {"messages": [response]}# add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", END)

# Compile the state graph into a runnable object
graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})