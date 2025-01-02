import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()


st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]
# st write magic
"""
In this example, we're going to be creating our own [`BaseCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html) called StreamHandler 
to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/) invocations and leveraging callbacks in our 
graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

The BaseCallBackHandler is a [Mixin](https://www.wikiwand.com/en/articles/Mixin) overloader function which we will use
to implement only `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model.

--- 
"""

openai_api_key = st.secrets["openai_api_key"]

# Title of the app
st.title("Project-Based Learning Design Assistant")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)


if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]


# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        # create a new container for streaming messages only, and give it context
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        # Add that last message to the st_message_state
        # Streamlit's refresh the message will automatically be visually rendered bc of the msg render for loop above
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))