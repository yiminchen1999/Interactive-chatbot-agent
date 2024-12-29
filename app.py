import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from graph import invoke_our_graph
from app_ver1 import get_streamlit_cb

# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize as an empty list
if "intake" not in st.session_state:
    st.session_state["intake"] = {}  # Initialize intake responses
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "intake_questions"

# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]
# Title of the app
st.title("Project-Based Learning Design Assistant")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Utility function to get a Streamlit callback handler with context

load_dotenv()

st.title("Project-Based Learning Design Assistant")
st.markdown("#### Step-by-Step Design with Dynamic Feedback")

# Check if the API key is available as an environment variable
if not os.getenv('API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    api_key = st.secrets["openai_api_key"]
    os.environ["API_KEY"] = api_key
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your API_KEY in the sidebar.")
        st.stop()

if "messages" not in st.session_state:
    # Default initial message to render in the message state
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. How can I assist you?")]

# Loop through all messages in the session state and render them as a chat on every refresh
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handles new user input and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handle graph events using the callback mechanism
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()  # Placeholder for dynamically updating AI's response
        st_callback = get_streamlit_cb(st.empty())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))  # Add the last message to the message state
        msg_placeholder.write(last_msg)  # Visually refresh the response after processing
