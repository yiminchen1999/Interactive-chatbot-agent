
import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize as an empty list

# Streamlit page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Title of the app
st.title("LangGraph Chatbot")
openai_api_key = "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"
# Initialize the OpenAI LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)


# Define the state as a TypedDict
class State(TypedDict):
    messages: list  # List of tuples (sender, message)


# Define the chatbot function
def chatbot(state: State) -> State:
    user_message = state["messages"][-1][1]  # Last user message
    response = llm.invoke([{"role": "user", "content": user_message}])
    state["messages"].append(("assistant", response.content))  # Extract content from the response
    return state


# Initialize StateGraph using the State type
graph_builder = StateGraph(State)

# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()


# Function to stream graph updates
def stream_graph_updates(user_input: str):
    initial_state: State = {"messages": [("user", user_input)]}
    updated_state = graph.invoke(initial_state)
    return [msg[1] for msg in updated_state["messages"] if msg[0] == "assistant"]


# Sidebar for user input
def chatbot_sidebar():
    st.sidebar.title("Chat with the Assistant")
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")
    if st.sidebar.button("Send"):
        if user_input:
            # Add user message to conversation history
            st.session_state["messages"].append(("user", user_input))

            # Get chatbot response
            responses = stream_graph_updates(user_input)

            # Add chatbot responses to conversation history
            for response in responses:
                st.session_state["messages"].append(("assistant", response))


# Main page to display conversation history
def display_chat():
    st.write("### Conversation:")
    # Safely access messages
    messages = st.session_state.get("messages", [])
    for sender, message in messages:
        if sender == "user":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Assistant:** {message}")


# Run the sidebar and main chat display
chatbot_sidebar()
display_chat()

