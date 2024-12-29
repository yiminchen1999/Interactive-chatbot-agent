import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]
# Title of the app
st.title("Project-Based Learning Design Assistant")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. How can I assist you?")]
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "intake_questions"  # Initial step
if "node_states" not in st.session_state:
    st.session_state["node_states"] = {}  # Track states for each node

# Loop through all messages in the session state and render them as a chat on every refresh
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handles new user input and invokes the graph node-by-node
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the current step based on the graph
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()  # Placeholder for dynamically updating AI's response
        st_callback = get_streamlit_cb(st.empty())

        # Prepare the current state and invoke the graph
        node_state = st.session_state["node_states"].get(st.session_state["current_step"], [])
        node_state.append(HumanMessage(content=prompt))

        # Invoke the current graph step
        response = invoke_our_graph(node_state, [st_callback], current_node=st.session_state["current_step"])

        # Extract the assistant's message
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))
        msg_placeholder.write(last_msg)

        # Save the updated state for the node
        st.session_state["node_states"][st.session_state["current_step"]] = response["messages"]

        # Advance to the next step based on the flow
        if "next_node" in response:
            st.session_state["current_step"] = response["next_node"]

