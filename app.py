import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize as an empty list
if "intake" not in st.session_state:
    st.session_state["intake"] = {}  # Initialize intake responses

# Streamlit page configuration
st.set_page_config(page_title="PBL Design Assistant", page_icon="📚")
openai_api_key = st.secrets["openai_api_key"]
# Title of the app
st.title("Project-Based Learning Design Assistant")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Define the state as a TypedDict
class State(TypedDict):
    messages: list  # List of tuples (sender, message)
    intake: dict  # Intake responses

# Define functions for each step in the PBL process
def intake_questions(state: State) -> State:
    """Collects teacher-specific context for the PBL design process."""
    intake_questions = [
        "In which state and district do you teach?",
        "Which grade level and subject area(s) do you teach?",
        "What is the topic for your project?",
        "Which set of content standards will you be using (e.g., Common Core, NGSS, state-level standards)?",
        "Are there specific skills you want students to develop (e.g., social-emotional learning, 21st-century skills)?",
        "How long should the project last?",
        "How long are your class periods?",
        "Do you want the students to work in groups?",
        "What types of technology do the students have access to?",
        "Is there a specific pedagogical model you would like to follow (e.g., Understanding by Design)?"
    ]
    for question in intake_questions:
        if question not in state["intake"]:
            state["messages"].append(("assistant", question))
            return state
    return state

def generate_project_idea(state: State) -> State:
    """Generates a draft project idea based on intake responses."""
    intake_summary = "\n".join(f"{key}: {value}" for key, value in state["intake"].items())
    prompt = f"Using the following context, generate a one-paragraph project idea:\n{intake_summary}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["messages"].append(("assistant", response.content))
    return state

def refine_project_idea(state: State) -> State:
    """Refines the project idea based on user feedback."""
    user_message = state["messages"][-1][1]
    prompt = f"Refine the project idea based on this feedback: {user_message}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["messages"].append(("assistant", response.content))
    return state

def generate_driving_questions(state: State) -> State:
    """Generates three draft driving questions."""
    project_idea = state["messages"][-1][1]
    prompt = f"Based on the project idea: {project_idea}, generate three draft driving questions."
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["messages"].append(("assistant", response.content))
    return state

def finalize_output(state: State) -> State:
    """Finalizes the project idea and driving questions for download."""
    project_idea = state["messages"][-2][1]
    driving_questions = state["messages"][-1][1]
    final_output = f"Project Idea:\n{project_idea}\n\nDriving Questions:\n{driving_questions}"
    state["messages"].append(("assistant", "Your project idea and driving questions are ready for download."))
    state["output"] = final_output
    return state

# Initialize StateGraph using the State type
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("intake_questions", intake_questions)
graph_builder.add_node("generate_project_idea", generate_project_idea)
graph_builder.add_node("refine_project_idea", refine_project_idea)
graph_builder.add_node("generate_driving_questions", generate_driving_questions)
graph_builder.add_node("finalize_output", finalize_output)

# Define the flow
graph_builder.set_entry_point("intake_questions")
graph_builder.add_edge("intake_questions", "generate_project_idea")
graph_builder.add_edge("generate_project_idea", "refine_project_idea")
graph_builder.add_edge("refine_project_idea", "generate_driving_questions")
graph_builder.add_edge("generate_driving_questions", "finalize_output")
graph_builder.set_finish_point("finalize_output")

# Compile the graph
graph = graph_builder.compile()

# Sidebar for teacher intake questions
def intake_sidebar():
    st.sidebar.title("Teacher Intake")
    intake_prompts = [
        ("state_district", "In which state and district do you teach?"),
        ("grade_subject", "Which grade level and subject area(s) do you teach?"),
        ("topic", "What is the topic for your project?"),
        ("standards", "Which set of content standards will you be using?"),
        ("skills", "Specific skills for students to develop?"),
        ("duration", "How long should the project last?"),
        ("class_periods", "How long are your class periods?"),
        ("group_work", "Do you want the students to work in groups?"),
        ("technology", "What types of technology do students have access to?"),
        ("pedagogical_model", "Specific pedagogical model to follow?")
    ]

    for key, prompt in intake_prompts:
        response = st.sidebar.text_input(prompt, key=key)
        if response:
            st.session_state["intake"][key] = response

# Sidebar for user input
def chatbot_sidebar():
    st.sidebar.title("Chat with the Assistant")
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")
    if st.sidebar.button("Send"):
        if user_input:
            # Add user message to conversation history
            st.session_state["messages"].append(("user", user_input))

            # Process user input through the graph
            initial_state: State = {"messages": st.session_state["messages"], "intake": st.session_state["intake"]}
            updated_state = graph.invoke(initial_state)

            # Update session state with responses
            st.session_state["messages"] = updated_state["messages"]
            st.session_state["intake"] = updated_state.get("intake", {})

# Main app display
def display_chat():
    st.write("### Conversation:")
    messages = st.session_state.get("messages", [])
    for sender, message in messages:
        if sender == "user":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Assistant:** {message}")

    if "output" in st.session_state:
        st.download_button("Download Final Output", st.session_state["output"], "final_output.txt")

# Run the sidebar and main chat display
intake_sidebar()
chatbot_sidebar()
display_chat()
