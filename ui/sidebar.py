import streamlit as st
import uuid
import textwrap
from database.db import (
    retrieve_all_titles,
    save_conversation,
    delete_conversation
)
from agent.graph import chatbot

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    new_id = generate_thread_id()
    save_conversation(new_id)
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    st.rerun()


def load_messages(thread_id: str):
    """Loads historical messages directly from LangGraph SQLite checkpointer state."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def _inject_sidebar_css():
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --vm-bg: #0E1116;
            --vm-bg-elevated: #171B21;
            --vm-bg-hover: #1D232B;
            --vm-border: rgba(255,255,255,0.07);
            --vm-text: #E7E9EC;
            --vm-text-muted: #838C99;
            --vm-accent: #34D399;
            --vm-accent-soft: rgba(52,211,153,0.14);
            --vm-danger: #F87171;
        }

        /* Sidebar shell */
        section[data-testid="stSidebar"] {
            background: var(--vm-bg);
            border-right: 1px solid var(--vm-border);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 0.6rem;
        }
        section[data-testid="stSidebar"] * {
            color: var(--vm-text);
        }

        /* Scrollbar */
        section[data-testid="stSidebar"] ::-webkit-scrollbar { width: 6px; }
        section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background: #2A313B; border-radius: 6px;
        }

        /* Logo header */
        .vm-logo-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 4px 14px 4px;
        }
        .vm-logo-text { display: flex; flex-direction: column; line-height: 1.05; }
        .vm-logo-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.18rem;
            letter-spacing: -0.01em;
            color: var(--vm-text);
        }
        .vm-logo-sub {
            font-size: 0.66rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--vm-accent);
            margin-top: 2px;
        }

        .vm-divider {
            height: 1px;
            background: var(--vm-border);
            margin: 6px 2px 14px 2px;
            border: none;
        }

        .vm-section-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--vm-text-muted);
            padding: 0 6px 8px 6px;
        }

        .vm-empty {
            color: var(--vm-text-muted);
            font-size: 0.83rem;
            padding: 10px 8px;
            border: 1px dashed var(--vm-border);
            border-radius: 10px;
            margin: 4px 2px;
        }

        /* Generic sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 9px;
            text-align: left;
            font-size: 0.87rem;
            padding: 0.42rem 0.6rem;
            transition: background 0.12s ease, border-color 0.12s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: var(--vm-bg-hover);
            border-color: var(--vm-border);
            color: var(--vm-text);
        }
        section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
            box-shadow: none;
        }

        /* New chat button — stands out as a filled pill */
        div[class*="st-key-new_chat_btn"] .stButton > button {
            background: var(--vm-accent-soft);
            border: 1px solid rgba(52,211,153,0.35);
            color: var(--vm-accent);
            font-weight: 600;
            border-radius: 10px;
            text-align: center;
        }
        div[class*="st-key-new_chat_btn"] .stButton > button:hover {
            background: rgba(52,211,153,0.22);
            border-color: var(--vm-accent);
        }

        /* Active conversation row */
        div[class*="st-key-active_"] .stButton > button {
            background: var(--vm-bg-elevated);
            border-left: 2px solid var(--vm-accent);
            border-radius: 8px;
            color: var(--vm-text);
            font-weight: 500;
        }

        /* Delete (trash) buttons — quiet by default */
        div[class*="st-key-del_"] .stButton > button {
            background: transparent;
            border: none;
            color: var(--vm-text-muted);
            padding: 0.42rem 0.3rem;
            font-size: 0.82rem;
        }
        div[class*="st-key-del_"] .stButton > button:hover {
            color: var(--vm-danger);
            background: rgba(248,113,113,0.1);
        }
        </style>
        """
    st.markdown(textwrap.dedent(css).strip(), unsafe_allow_html=True)


def _logo_svg() -> str:
    return (
        '<svg width="34" height="34" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="40" height="40" rx="10" fill="#12171D"/>'
        '<path d="M11 11c0 7 6 7.5 6 12.5S13 28 13 28" stroke="#34D399" stroke-width="2.3" '
        'stroke-linecap="round" fill="none"/>'
        '<path d="M29 11c0 7-6 7.5-6 12.5S27 28 27 28" stroke="#5EEAD4" stroke-width="2.3" '
        'stroke-linecap="round" fill="none"/>'
        '<circle cx="11" cy="10" r="1.7" fill="#34D399"/>'
        '<circle cx="29" cy="10" r="1.7" fill="#5EEAD4"/>'
        '<circle cx="20" cy="29" r="2.1" fill="#34D399"/>'
        '</svg>'
    )

def render_sidebar():
    _inject_sidebar_css()

    logo_html = (
        f'<div class="vm-logo-row">{_logo_svg()}'
        '<div class="vm-logo-text">'
        '<span class="vm-logo-title">VenoMind</span>'
        '<span class="vm-logo-sub">AI Assistant</span>'
        '</div></div>'
    )
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)

    with st.sidebar.container(key="new_chat_btn"):
        if st.button("＋  New chat", use_container_width=True):
            reset_chat()

    st.sidebar.markdown('<hr class="vm-divider"/>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="vm-section-label">Recent</div>', unsafe_allow_html=True)

    titles = retrieve_all_titles()

    if not titles:
        st.sidebar.markdown(
            '<div class="vm-empty">No conversations yet — start one above.</div>',
            unsafe_allow_html=True,
        )
        return

    for thread_id, title in titles.items():
        is_active = (thread_id == st.session_state.get("thread_id"))
        raw_title = title if title else "New Conversation"
        truncated_title = raw_title[:24] + "…" if len(raw_title) > 24 else raw_title

        row_key = f"active_{thread_id}" if is_active else f"chat_{thread_id}"

        with st.sidebar.container(key=row_key):
            col1, col2 = st.columns([0.83, 0.17])

            icon = "👉" if is_active else "💬"
            if col1.button(f"{icon} {truncated_title}", key=f"btn_{thread_id}", use_container_width=True):
                st.session_state["thread_id"] = thread_id
                raw_messages = load_messages(thread_id)
                st.session_state["message_history"] = [
                    {
                        "role": "user" if msg.type == "human" else "assistant",
                        "content": msg.content,
                    }
                    for msg in raw_messages
                ]
                st.rerun()

            with col2:
                with st.container(key=f"del_{thread_id}"):
                    if st.button("✕", key=f"del_btn_{thread_id}", help="Delete conversation"):
                        delete_conversation(thread_id)
                        if st.session_state.get("thread_id") == thread_id:
                            st.session_state["thread_id"] = None
                            st.session_state["message_history"] = []
                        st.rerun()