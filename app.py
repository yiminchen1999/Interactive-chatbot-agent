import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context



st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
# st write magic
"""

"""


# Title of the app
st.title("Project-Based Learning Design Assistant")
# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. How can I assist you?")]

# Render the conversation history
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle new user input
user_input = st.chat_input("Your message:") if hasattr(st, "chat_input") else st.text_input("Your message:")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Process the AI's response
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        st_callback = get_streamlit_cb(st.container())

        try:
            response = invoke_our_graph(st.session_state.messages, [st_callback])
            last_msg = response["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=last_msg))
            msg_placeholder.write(last_msg)
        except Exception as e:
            st.error(f"An error occurred: {e}")