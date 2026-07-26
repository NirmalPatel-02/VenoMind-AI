import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import chatbot, generate_title
from database.db import update_title, retrieve_all_titles

def render_chat_interface():
    thread_id = st.session_state.get("thread_id")
    
    if not thread_id:
        st.info("Start a new chat from the sidebar to begin!")
        return

    for message in st.session_state.get("message_history", []):
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
            response_stream = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )
            
            def stream_parser():
                for message_chunk, metadata in response_stream:
                    if message_chunk.content:
                        yield message_chunk.content

            ai_message = st.write_stream(stream_parser())

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        st.rerun()