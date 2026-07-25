import streamlit as st
from pathlib import Path
import sys
import uuid
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from backend.backend import chatbot , model
from langchain_core.messages import HumanMessage , SystemMessage


## functions

def generate_thread_id():
    id = uuid.uuid4()
    return id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state["conv_name"][thread_id] = "New Conversation"
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

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

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'conv_name' not in st.session_state:
    st.session_state['conv_name'] = {}

add_thread(st.session_state['thread_id'])

if st.session_state['thread_id'] not in st.session_state['conv_name']:
    st.session_state['conv_name'][st.session_state['thread_id']] = "New Conversation"

## sidebar code


st.sidebar.title('VenoMind AI')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Conversation History')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(st.session_state["conv_name"][thread_id],key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_messages(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message , HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role':role , 'content': message.content})  
        st.session_state['message_history'] = temp_messages  



for message in  st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
 
user_input = st.chat_input("Type Here")

if user_input:

    st.session_state['message_history'].append({'role':'user' , 'content':user_input})

    with st.chat_message('user'):
        st.text(user_input)

    thread_id = st.session_state["thread_id"]

    if st.session_state["conv_name"][thread_id] == "New Conversation":
        title = generate_title(user_input)
        st.session_state["conv_name"][thread_id] = title

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