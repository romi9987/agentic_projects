import streamlit as st
from typing import TypedDict, List, Literal
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from redis_helper import (
    save_state,
    load_state,
    archive_chat,
    show_archive,
    delete_all_archives,
)

DEVELOPER_MODE = False  # set to False before demo
SESSION_ID = "default"  # later you can tie this to user login, or a random ID


# Define State - a dictionary with history, user_input, response.
class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatState(TypedDict):
    summary: str
    history: List[ChatMessage]  # recent turns only (for LLM)
    display_history: List[ChatMessage]  # full chat log (for UI)
    user_input: str
    response: str


# Initialize LLM
llm = OllamaLLM(model="llama3")  # local model

# Prompt Template with conversation history
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n"
            "You have access to a private summary of the previous conversation:\n{summary}\n"
            "Do not reveal or mention this summary to the user.",
        ),
        (
            "system",
            "Here is the recent conversation between the user and the assistant:\n{history}",
        ),
        ("human", "{user_input}"),
    ]
)

summary_prompt = ChatPromptTemplate.from_template(
    """
    You are an **internal memory manager**.
    Your task is ONLY to update the hidden summary of the conversation.
    DO NOT generate dialogue or answer questions.
    Keep it concise (max 5 sentences), capturing only key facts.

    Current summary:
    {summary}

    New exchange:
    USER: {user_input}
    ASSISTANT: {response}

    Updated summary:
    """
)

# Linking a prompt to an LLM within a chain
chat_chain = chat_prompt | llm
summary_chain = summary_prompt | llm


# Graph node (chatbot_node) = runs prompt | llm, updates history.
# --- Node function (how each step works) ---
def chatbot_node(state: ChatState):
    # Join history into one string
    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in state["history"]
    )
    # Generate assistant response
    response = chat_chain.invoke(
        {
            "summary": state["summary"],
            "history": history_text,
            "user_input": state["user_input"],
        }
    )

    # Update rolling history (keep only last 4 exchanges)
    # --- Internal history (truncate) ---
    new_history = state["history"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response},
    ]
    if len(new_history) > 8:  # 4 turns (user+assistant)
        new_history = new_history[-8:]

    # --- Display history (never truncated) ---
    new_display_history = state["display_history"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response},
    ]

    # --- Update summary every 3 exchanges ---
    turn_count = len(new_display_history) // 2  # count user+assistant pairs
    if turn_count % 3 == 0:
        new_summary = summary_chain.invoke(
            {
                "summary": state["summary"],
                "user_input": state["user_input"],
                "response": response,
            }
        )
    else:
        new_summary = state["summary"]

    return {
        "summary": new_summary,
        "history": new_history,
        "display_history": new_display_history,
        "response": response,
    }


# Graph = entry → chatbot → END.
# --- Build graph ---
graph = StateGraph(ChatState)
# Pass update_every dynamically via lambda
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

# Compile the graph into an app
app = graph.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI Chatbot")
st.write(
    "Mini Agent uruchomiony. Napisz 'exit' aby zakończyć rozmowę albo 'new' aby otworzyć nowy czat."
)

if DEVELOPER_MODE:
    # Sidebar controls
    debug_mode = st.sidebar.checkbox("🔍 Debug mode")
    update_every = st.sidebar.slider("🧠 Update summary every N turns", 1, 10, 3, 1)
    st.session_state.update_every = update_every
else:
    debug_mode = False
    st.session_state.update_every = 3  # default

if "graph_state" not in st.session_state:
    st.session_state.graph_state = load_state(SESSION_ID)

# Display conversation (full log for user)
for msg in st.session_state.graph_state["display_history"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Debug view (hidden summary)
if debug_mode:
    with st.sidebar.expander("🧠 Hidden Summary", expanded=True):
        st.markdown(st.session_state.graph_state["summary"] or "_(empty)_")

# Handle input
user_input = st.chat_input("Zadaj pytanie:")
if user_input and user_input.strip():
    if user_input.lower() in ["exit", "quit"]:
        archive_chat(SESSION_ID, st.session_state.graph_state)
        st.session_state.graph_state = {
            "summary": "",
            "history": [],
            "display_history": [],
        }
        st.warning("Rozmowa zakończona.")
        st.stop()
    elif user_input.lower() in "new":
        archive_chat(SESSION_ID, st.session_state.graph_state)
        st.session_state.graph_state = {
            "summary": "",
            "history": [],
            "display_history": [],
        }
        st.rerun()

    # --- Show user's message immediately ---
    st.chat_message("user").markdown(user_input)

    # Update state via LangGraph
    st.session_state.graph_state["user_input"] = user_input

    # Run LangGraph app
    result = app.invoke(st.session_state.graph_state)

    # Update Streamlit memory
    st.session_state.graph_state = result
    save_state(SESSION_ID, result)

    # Show assistant response
    st.chat_message("assistant").markdown(result["response"])

# --- Sidebar: archives ---
with st.sidebar.expander("📚 Archived Conversations", expanded=False):
    show_archive(SESSION_ID)

if DEVELOPER_MODE:
    with st.sidebar:
        if st.button("🗑️ Delete all archives"):
            delete_all_archives(SESSION_ID)
            st.session_state.graph_state = {
                "summary": "",
                "history": [],
                "display_history": [],
            }
            st.success("All chats deleted. Starting from scratch.")
            st.rerun()
