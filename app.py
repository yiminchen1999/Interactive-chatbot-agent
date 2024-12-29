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
    st.session_state["current_step"] = "intake_questions"

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
            if state["messages"] and state["messages"][-1][0] == "user":
                state["intake"][question] = state["messages"][-1][1]
                state["messages"].append(("assistant", f"Got it! Moving to the next question."))
            else:
                state["messages"].append(("assistant", question))
            return state

    st.session_state["current_step"] = "generate_project_idea"
    return state

def generate_project_idea(state: State) -> State:
    """Generates a draft project idea based on intake responses."""
    if "project_idea" not in st.session_state:
        intake_summary = "\n".join(f"{key}: {value}" for key, value in state["intake"].items())
        prompt = f"Using the following context, generate a one-paragraph project idea:\n{intake_summary}"
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["messages"].append(("assistant", response.content))
        st.session_state["project_idea"] = response.content
    st.session_state["current_step"] = "refine_project_idea"
    return state

def refine_project_idea(state: State) -> State:
    """Refines the project idea based on user feedback."""
    if state["messages"] and state["messages"][-1][0] == "user":
        user_feedback = state["messages"][-1][1]
        prompt = f"Refine the project idea based on this feedback: {user_feedback}"
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["messages"].append(("assistant", response.content))
        st.session_state["current_step"] = "generate_driving_questions"
    else:
        state["messages"].append(("assistant", "Provide feedback on the project idea:"))
    return state

def generate_driving_questions(state: State) -> State:
    """Generates three draft driving questions."""
    if "driving_questions" not in st.session_state:
        project_idea = st.session_state["project_idea"]
        prompt = f"Based on the project idea: {project_idea}, generate three draft driving questions."
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["messages"].append(("assistant", response.content))
        st.session_state["driving_questions"] = response.content
    st.session_state["current_step"] = "refine_driving_questions"
    return state

def refine_driving_questions(state: State) -> State:
    """Refines the driving questions based on user feedback."""
    if state["messages"] and state["messages"][-1][0] == "user":
        user_feedback = state["messages"][-1][1]
        prompt = f"Refine the driving questions based on this feedback: {user_feedback}"
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["messages"].append(("assistant", response.content))
        st.session_state["current_step"] = "finalize_output"
    else:
        state["messages"].append(("assistant", "Provide feedback on the driving questions:"))
    return state

def finalize_output(state: State) -> State:
    """Finalizes the project idea and driving questions for download."""
    if "final_output" not in st.session_state:
        project_idea = st.session_state["project_idea"]
        driving_questions = st.session_state["driving_questions"]
        final_output = f"Project Idea:\n{project_idea}\n\nDriving Questions:\n{driving_questions}"
        state["messages"].append(("assistant", "Your project idea and driving questions are ready for download."))
        st.session_state["final_output"] = final_output
    return state

# Main app logic based on the current step
def run_current_step():
    state: State = {"messages": st.session_state["messages"], "intake": st.session_state["intake"]}

    if st.session_state["current_step"] == "intake_questions":
        updated_state = intake_questions(state)
    elif st.session_state["current_step"] == "generate_project_idea":
        updated_state = generate_project_idea(state)
    elif st.session_state["current_step"] == "refine_project_idea":
        updated_state = refine_project_idea(state)
    elif st.session_state["current_step"] == "generate_driving_questions":
        updated_state = generate_driving_questions(state)
    elif st.session_state["current_step"] == "refine_driving_questions":
        updated_state = refine_driving_questions(state)
    elif st.session_state["current_step"] == "finalize_output":
        updated_state = finalize_output(state)
    else:
        updated_state = state

    # Update session state with responses
    st.session_state["messages"] = updated_state["messages"]
    st.session_state["intake"] = updated_state.get("intake", {})

# Sidebar for user input
def chatbot_sidebar():
    st.sidebar.title("Chat with the Assistant")
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")
    if st.sidebar.button("Send"):
        if user_input:
            # Add user message to conversation history
            st.session_state["messages"].append(("user", user_input))
            run_current_step()

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

# Main app display
def display_chat():
    st.write("### Conversation:")
    messages = st.session_state.get("messages", [])
    for sender, message in messages:
        if sender == "user":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Assistant:** {message}")

    if "final_output" in st.session_state:
        st.download_button("Download Final Output", st.session_state["final_output"], "final_output.txt")

# Run the app
intake_sidebar()
chatbot_sidebar()
display_chat()
