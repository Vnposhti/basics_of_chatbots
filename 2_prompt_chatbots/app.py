import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- Chatbot Functions ---

# Shared session history store
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history_func(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 1-PromptChatTemplate.py functionality
def prompt_chat_template_chatbot():
    st.title("Sarcastic Assistant Chatbot")
    st.write("Chat with a sarcastic AI.")

    llm = ChatGroq(model_name="llama3-8b-8192")
    outputparser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a sarcastic assistant."),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    chain = prompt | llm | outputparser
    with_message_history = RunnableWithMessageHistory(chain, get_session_history_func)

    session_id = st.text_input("Enter Session ID:", key="sarcastic_session_id")
    if not session_id:
        return

    config = {"configurable": {"session_id": session_id}}

    # Display existing messages
    if session_id in st.session_state.store:
        for msg in st.session_state.store[session_id].messages:
            if isinstance(msg, HumanMessage):
                st.text(f"You: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.text(f"AI: {msg.content}")

    user_input = st.text_input("Ask anything:", key="sarcastic_user_input")

    if st.button("Send", key="sarcastic_send_button") and user_input:
        with st.spinner("Thinking..."):
            response = with_message_history.invoke([HumanMessage(content=user_input)], config=config)
            st.text(response)
        # Rerun to update history display
        st.rerun()

    if st.button("Clear Session History", key="sarcastic_clear_history"):
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]
            st.success(f"Session history for ID '{session_id}' cleared.")
            st.rerun()

# 2-complex_prompts.py functionality
def complex_prompts_chatbot():
    st.title("Multi-language Assistant Chatbot")
    st.write("Chat with an AI that responds in a specified language.")

    llm = ChatGroq(model_name="llama3-70b-8192")
    outputparser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm | outputparser
    with_message_history = RunnableWithMessageHistory(chain, get_session_history_func, input_messages_key="messages")

    session_id = st.text_input("Enter Session ID:", key="complex_session_id")
    if not session_id:
        st.warning("Please enter a Session ID to start chatting.")
        return
    
    language = st.selectbox("Choose Response Language:", ("English", "Hindi", "Gujarati"), key="response_language")

    config = {"configurable": {"session_id": session_id}}

    # Display existing messages
    if session_id in st.session_state.store:
        for msg in st.session_state.store[session_id].messages:
            if isinstance(msg, HumanMessage):
                st.text(f"You: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.text(f"AI: {msg.content}")

    user_input = st.text_input("Ask anything:", key="complex_user_input")

    if st.button("Send", key="complex_send_button") and user_input:
        with st.spinner("Thinking..."):
            response = with_message_history.invoke(
                {
                    'messages': [HumanMessage(content=user_input)],
                    "language": language
                }, 
                config=config
            )
            st.text(response)
        # Rerun to update history display
        st.rerun()

    if st.button("Clear Session History", key="complex_clear_history"):
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]
            st.success(f"Session history for ID '{session_id}' cleared.")
            st.rerun()

# --- Main Streamlit App --- 
st.sidebar.title("Chatbot Options")

selected_chatbot = st.sidebar.selectbox(
    "Choose a Chatbot:",
    (
        "Sarcastic Assistant",
        "Multi-language Assistant"
    )
)

if selected_chatbot == "Sarcastic Assistant":
    prompt_chat_template_chatbot()
elif selected_chatbot == "Multi-language Assistant":
    complex_prompts_chatbot()
