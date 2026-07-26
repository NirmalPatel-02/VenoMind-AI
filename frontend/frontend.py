import streamlit as st
from pathlib import Path
import sys
import uuid
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from backend.backend import chatbot , model , retrive_all_threads, retrive_all_titles, save_conversation, update_title
from langchain_core.messages import HumanMessage , SystemMessage


## functions

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    save_conversation(thread_id)
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

def load_messages(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def generate_title(first_message):
    response = model.invoke([
        SystemMessage(
            content="""
                Generate a very short conversation title.

                Rules:
                - Maximum 5 words.
                - Do not use quotes.
                - Do not add punctuation.
                - Only return the title.
                """
        ),
        HumanMessage(content=first_message)
    ])

    return response.content.strip()


if 'message_history' not in st.session_state: 
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    threads = retrive_all_threads()
    if len(threads) == 0:
        thread_id = generate_thread_id()
        save_conversation(thread_id)

        st.session_state["thread_id"] = thread_id

    else:
        st.session_state["thread_id"] = threads[-1]

messages = load_messages(st.session_state["thread_id"])

st.session_state["message_history"] = [
    {
        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
        "content": msg.content,
    }
    for msg in messages
]


## sidebar code


st.sidebar.title('VenoMind AI')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Conversation History')

titles = retrive_all_titles()

for thread_id, title in titles.items():
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_messages(thread_id)
        temp_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append({"role": role,"content": message.content})
        st.session_state["message_history"] = temp_messages

for message in  st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
 
user_input = st.chat_input("Type Here")

if user_input:

    st.session_state['message_history'].append({'role':'user' , 'content':user_input})

    with st.chat_message('user'):
        st.text(user_input)

    thread_id = st.session_state["thread_id"]

    titles = retrive_all_titles()

    if titles[thread_id] == "New Conversation":
        title = generate_title(user_input)
        update_title(thread_id, title)

    CONFIG = {'configurable':{'thread_id':thread_id}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk , metadata in chatbot.stream(
                {'messages':[HumanMessage(content = user_input)]},
                config = CONFIG , 
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role':'assistant' , 'content':ai_message})