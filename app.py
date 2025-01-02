import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")

openai_api_key = st.secrets["openai_api_key"]
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Initialize ChatOpenAI
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. How can I assist you?")]

# Render chat messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Input handling
try:
    user_input = st.chat_input("Your message:") if hasattr(st, "chat_input") else st.text_input("Your message:")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").write(user_input)

        # Handle response with callback
        with st.chat_message("assistant"):
            st_callback = get_streamlit_cb(st.container())
            response = invoke_our_graph(st.session_state.messages, [st_callback])
            last_msg = response["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=last_msg))
            st.write(last_msg)
except Exception as e:
    st.error(f"An error occurred: {e}")
