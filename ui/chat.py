import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import chatbot, generate_title
from database.db import update_title, retrieve_all_titles


def load_clean_message_history(thread_id: str):
    """Retrieves raw messages from checkpointer state and returns ONLY the final

    user prompt and final assistant response per turn.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config)
        raw_messages = state.values.get("messages", [])
    except Exception:
        raw_messages = []

    clean_history = []
    
    for msg in raw_messages:
        msg_type = getattr(msg, "type", None)
        content = getattr(msg, "content", "")

        if msg_type == "human" and content:
            clean_history.append({"role": "user", "content": content})

        elif msg_type == "ai" and content and not getattr(msg, "tool_calls", None):
            if clean_history and clean_history[-1]["role"] == "assistant":
                clean_history[-1]["content"] = content
            else:
                clean_history.append({"role": "assistant", "content": content})

    return clean_history


def render_chat_interface():
    thread_id = st.session_state.get("thread_id")

    if not thread_id:
        st.info("Start a new chat from the sidebar to begin!")
        return

    if "current_thread_id" not in st.session_state or st.session_state["current_thread_id"] != thread_id:
        st.session_state["current_thread_id"] = thread_id
        st.session_state["message_history"] = load_clean_message_history(thread_id)

    if "message_history" not in st.session_state:
        st.session_state["message_history"] = load_clean_message_history(thread_id)

    for message in st.session_state["message_history"]:
        if message.get("role") in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_input := st.chat_input("Ask VenoMind AI anything..."):

        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        titles = retrieve_all_titles()
        if titles.get(thread_id) == "New Conversation":
            new_title = generate_title(user_input)
            update_title(thread_id, new_title)

        config = {"configurable": {"thread_id": thread_id}}

        with st.chat_message("assistant"):
            with st.status("🧠 VenoMind AI is processing...", expanded=True) as status_box:
                final_response_text = ""

                events = chatbot.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "query": user_input,
                        "iteration": 0,
                        "max_iteration": 2,
                    },
                    config=config,
                    stream_mode="updates",
                )

                for event in events:
                    for node_name, node_update in event.items():

                        if node_name == "chat_node":
                            messages = node_update.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if getattr(last_msg, "tool_calls", None):
                                    status_box.update(
                                        label="🌐 Searching live web sources via DuckDuckGo...",
                                        state="running",
                                    )
                                else:
                                    final_response_text = node_update.get("answer", last_msg.content)

                        elif node_name == "tools":
                            messages = node_update.get("messages", [])
                            if messages:
                                snippet = str(messages[-1].content)[:250]
                                st.write("🔍 **Retrieved Search Context:**")
                                st.caption(f"\"{snippet}...\"")
                                status_box.update(
                                    label="⚡ Synthesizing retrieved data...",
                                    state="running",
                                )

                        elif node_name == "evaluate_answer":
                            eval_status = node_update.get("evaluation")
                            feedback = node_update.get("feedback")

                            if eval_status == "approved":
                                status_box.update(
                                    label="✅ Quality check passed!",
                                    state="complete",
                                    expanded=False,
                                )
                            else:
                                status_box.update(
                                    label="🔍 QA Evaluator flagged draft. Optimizing response...",
                                    state="running",
                                    expanded=False,
                                )
                                st.warning(f"**Evaluator Critique:** {feedback}")

                        elif node_name == "optimize_answer":
                            final_response_text = node_update.get("answer", "")
                            status_box.update(
                                label="✨ Answer optimized and verified!",
                                state="complete",
                                expanded=False,
                            )

            if final_response_text:
                st.markdown(final_response_text)
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": final_response_text}
                )

        st.rerun()