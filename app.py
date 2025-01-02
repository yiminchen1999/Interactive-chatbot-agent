import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

# Load environment variables


# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]

# Title of the app
st.title("Project-Based Learning Design Assistant")

# Define state for LangGraph
class GraphsState(TypedDict):
    messages: list

# Initialize LangGraph state graph
graph = StateGraph(GraphsState)

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)
    response = llm.invoke(messages)
    return {"messages": [response]}  # Add the response to the messages using LangGraph reducer paradigm

# Define the structure of the graph
graph.add_edge("START", "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", "END")

# Compile the graph
graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Welcome! Let’s get started. Can you type your intake here?")]

# Loop through all messages in the session state and render them as a chat on every refresh
for msg in st.session_state["messages"]:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Takes new input in chat box from user and invokes the graph step-by-step
try:
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("Your message:")
        if user_input:
            st.session_state["messages"].append(HumanMessage(content=user_input))
            st.chat_message("user").write(user_input)

            # Process the AI's response and handle graph events using the callback mechanism
            with st.chat_message("assistant"):
                msg_placeholder = st.empty()  # Placeholder for dynamically updating AI's response
                st_callback = get_streamlit_cb(st.container())

                # Invoke the graph with user input and the callback
                response = invoke_our_graph(st.session_state["messages"], [st_callback])

                # Extract the assistant's response and append it to the session state
                last_msg = response["messages"][-1].content
                st.session_state["messages"].append(AIMessage(content=last_msg))
                msg_placeholder.write(last_msg)
    else:
        user_input = st.text_input("Your message:", key="input", placeholder="Type your message here...")
        if st.button("Send") and user_input.strip():
            st.session_state["messages"].append(HumanMessage(content=user_input))
            st.write(f"**You:** {user_input}")

            # Process the AI's response and handle graph events using the callback mechanism
            with st.container():
                msg_placeholder = st.empty()  # Placeholder for dynamically updating AI's response
                st_callback = get_streamlit_cb(st.container())

                # Invoke the graph with user input and the callback
                response = invoke_our_graph(st.session_state["messages"], [st_callback])

                # Extract the assistant's response and append it to the session state
                last_msg = response["messages"][-1].content
                st.session_state["messages"].append(AIMessage(content=last_msg))
                msg_placeholder.write(last_msg)
except Exception as e:
    st.error(f"An error occurred: {e}")

