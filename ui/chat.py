import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import chatbot, generate_title
from database.db import update_title, retrieve_all_titles


def load_clean_message_history(thread_id: str):
    """Retrieves raw messages from the chatbot checkpointer state and filters

    out internal tool call triggers & intermediate tool outputs for a clean UI.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config)
        raw_messages = state.values.get("messages", [])
    except Exception:
        raw_messages = []

    clean_history = []
    for msg in raw_messages:
        if getattr(msg, "type", None) == "human":
            clean_history.append({"role": "user", "content": msg.content})
        elif getattr(msg, "type", None) == "ai" and msg.content:
            clean_history.append({"role": "assistant", "content": msg.content})

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
            status_container = st.empty()
            search_status = None

            response_stream = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )

            def stream_parser():
                nonlocal search_status

                for message_chunk, metadata in response_stream:
                    if metadata.get("langgraph_node") == "chat_node" and getattr(
                        message_chunk, "tool_calls", None
                    ):
                        if search_status is None:
                            search_status = status_container.status(
                                "🌐 Searching the web for real-time information...",
                                expanded=True,
                            )
                            search_status.write("🔍 Querying web sources...")

                    if metadata.get("langgraph_node") == "tools":
                        if search_status is not None:
                            if message_chunk.content:
                                search_status.markdown("**Found Sources / Snippets:**")
                                search_status.write(message_chunk.content)
                            search_status.update(
                                label="✅ Search completed. Generating response...",
                                state="complete",
                                expanded=False,
                            )

                    if metadata.get("langgraph_node") == "chat_node" and message_chunk.content:
                        if isinstance(message_chunk.content, str):
                            yield message_chunk.content
                        elif isinstance(message_chunk.content, list):
                            for block in message_chunk.content:
                                if isinstance(block, dict) and "text" in block:
                                    yield block["text"]

            ai_message = st.write_stream(stream_parser())

        if ai_message:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )

        st.rerun()