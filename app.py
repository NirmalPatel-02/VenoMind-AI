import streamlit as st
import uuid
from database.db import init_db, retrieve_all_threads, save_conversation
from ui.sidebar import render_sidebar, load_messages
from ui.chat import render_chat_interface

st.set_page_config(
    page_title="VenoMind AI",
    page_icon="assets/venomind_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

if "thread_id" not in st.session_state or st.session_state["thread_id"] is None:
    threads = retrieve_all_threads()
    if not threads:
        initial_id = str(uuid.uuid4())
        save_conversation(initial_id)
        st.session_state["thread_id"] = initial_id
    else:
        st.session_state["thread_id"] = threads[0]

if "message_history" not in st.session_state or not st.session_state["message_history"]:
    raw_messages = load_messages(st.session_state["thread_id"])
    st.session_state["message_history"] = [
        {
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
        }
        for msg in raw_messages
    ]

render_sidebar()
render_chat_interface()