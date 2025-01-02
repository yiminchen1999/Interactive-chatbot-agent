

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]

# Title of the app
st.title("Project-Based Learning Design Assistant")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. How can I assist you?")]

# Loop through all messages in the session state and render them as a chat on every refresh
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Takes new input in chat box from user and invokes the graph step-by-step
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handle graph events using the callback mechanism
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()  # Placeholder for dynamically updating AI's response
        st_callback = get_streamlit_cb(st.container())

        # Invoke the graph with user input and the callback
        response = invoke_our_graph(st.session_state.messages, [st_callback])

        # Extract the assistant's response and append it to the session state
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))
        msg_placeholder.write(last_msg)
