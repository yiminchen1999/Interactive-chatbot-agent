import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# Ensure session state is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize as an empty list
if "intake" not in st.session_state:
    st.session_state["intake"] = {}  # Initialize intake responses
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "intake_questions"  # Start with intake questions

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
    """Refines the project idea without user feedback."""
    project_idea = state["messages"][-1][1]
    prompt = f"Refine the following project idea:\n{project_idea}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["messages"].append(("assistant", response.content))
    return state

def generate_driving_questions(state: State) -> State:
    """Generates draft driving questions."""
    project_idea = state["messages"][-1][1]
    prompt = f"Generate three driving questions based on the following project idea:\n{project_idea}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["messages"].append(("assistant", response.content))
    return state

def finalize_output(state: State) -> State:
    """Finalizes the output for download."""
    project_idea = state["messages"][-3][1]
    driving_questions = state["messages"][-1][1]
    final_output = f"Project Idea:\n{project_idea}\n\nDriving Questions:\n{driving_questions}"
    state["messages"].append(("assistant", "Your project idea and driving questions are ready for download."))
    state["output"] = final_output
    return state

# Initialize StateGraph
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

# Main app logic
if st.session_state["current_step"]:
    try:
        initial_state: State = {"messages": st.session_state["messages"], "intake": st.session_state["intake"]}
        updated_state = graph.invoke(st.session_state["current_step"], initial_state)
        st.session_state["messages"] = updated_state["messages"]
        st.session_state["intake"] = updated_state.get("intake", {})
        next_step = graph.get_next_node(st.session_state["current_step"])
        st.session_state["current_step"] = next_step if next_step else None
    except AttributeError as e:
        st.error(f"Error: {e}")
        st.stop()

# Display conversation
st.write("### Conversation:")
for sender, message in st.session_state["messages"]:
    if sender == "user":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Assistant:** {message}")

# Provide download option for finalized output
if "output" in st.session_state:
    st.download_button("Download Final Output", st.session_state["output"], "final_output.txt")
