from secret_api_key import groq_api_key
import streamlit as st
from typing import Annotated # Import the Annotated class for type hints with additional metadata
from typing_extensions import TypedDict # Import the TypedDict class for defining custom typed dictionaries
from langgraph.graph import StateGraph # Import the StateGraph class for creating state graphs
from langgraph.graph.message import add_messages # Import the add_messages function for adding messages to a list
from langchain_groq import ChatGroq

# Streamlit page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Title of the app
st.title("LangGraph Chatbot")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview")

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a StateGraph instance
graph_builder = StateGraph(State)

# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add chatbot to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()

# Function to stream graph updates
def stream_graph_updates(user_input: str):
    initial_state = {"messages": [("user", user_input)]}
    responses = []
    for event in graph.stream(initial_state):
        for value in event.values():
            responses.append(value["messages"][-1].content)
    return responses

# Initialize session state to store conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Sidebar for user input
def chatbot_sidebar():
    st.sidebar.title("Chat with the Assistant")
    
    # Input box for user message on the sidebar
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")
    
    # Submit button in the sidebar
    if st.sidebar.button("Send"):
        submit_message(user_input)

# Function to submit the message and generate response
def submit_message(user_input):
    if user_input:
        # Append user input to conversation history
        st.session_state['messages'].append(f"You: {user_input}")
        
        # Get chatbot response
        responses = stream_graph_updates(user_input)
        
        # Append chatbot responses to conversation history
        for response in responses:
            st.session_state['messages'].append(f"Assistant: {response}")

# Main page for displaying chat history
def display_chat():
    st.write("### Conversation:")
    for message in st.session_state['messages']:
        st.write(message)

# Run the sidebar and main chat display
chatbot_sidebar()  # Input and submit button on the sidebar
display_chat()  # Display conversation on the main page