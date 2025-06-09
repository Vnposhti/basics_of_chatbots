import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# 1-simple_llm.py
def simple_llm_chatbot():
    st.title("Simple LLM Chatbot (Q&A)")
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name="llama3-8b-8192")
    
    user_input = st.text_input("Ask anything:", "Tell me about LLMs")

    if st.button("Answer"):
        response = llm.invoke(user_input)
        st.write(response.content)

# 2-enhanced_llm.py
def enhanced_llm_chatbot():
    st.title("Enhanced LLM Chatbot (Q&A)")
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user queries"),
            ("user", "Question:{question}")
        ]
    )

    def generate_response(question):
        llm = ChatGroq(model_name="llama3-8b-8192")
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer

    user_input = st.text_input("Ask anything:", "Tell me about LLMs")

    if st.button('Answer'):
        response = generate_response(user_input)
        st.write(response)

# 3-conversation.py
def conversation_chatbot():
    st.title("Conversation Chatbot")
    st.write("Write 'exit' to stop the conversation.")
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage

    llm = ChatGroq(model_name="llama3-8b-8192")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    def get_conversation_response(user_input):
        st.session_state.conversation_history.append(HumanMessage(content=user_input))
        response = llm.invoke(st.session_state.conversation_history)
        st.session_state.conversation_history.append(AIMessage(content=response.content))
        return response.content
    
    user_input = st.text_input("Ask anything:", "Hi", key="conversation_input")

    if st.button("Send", key="conversation_send") and user_input:
        if user_input.lower() == 'exit':
            st.session_state.conversation_history = []
        else:
            ai_response = get_conversation_response(user_input)
            st.write(ai_response)

    st.subheader("Conversation History")
    for msg in st.session_state.conversation_history:
        if isinstance(msg, HumanMessage):
            st.text(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            st.text(f"AI: {msg.content}")

# 4 and 5 - conversation_with_session_history.py
def session_history_with_conversation():
    st.title("Session History with Conversation")
    st.write("Enter a Session ID to start or resume a conversation. Write 'exit' to stop the conversation.")
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    llm = ChatGroq(model_name="llama3-8b-8192")

    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_session_history_func(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    with_message_history = RunnableWithMessageHistory(llm, get_session_history_func)

    session_id = st.text_input("Session ID:", key="session_id_input")
    config = {"configurable": {"session_id": session_id}}

    user_input = st.text_input("Ask anything:","Tell me about machine learning ", key="session_chat_input")

    if st.button("Send", key="session_chat_send") and user_input:
        if session_id:
            if user_input.lower() == 'exit':
                st.write("AI: Goodbye!")
                if session_id in st.session_state.store:
                    st.subheader(f"Full Conversation History for Session ID - {session_id}")
                    conversation_history = st.session_state.store[session_id].messages
                    for msg in conversation_history:
                        if isinstance(msg, HumanMessage):
                            st.text(f"Human: {msg.content}")
                        elif isinstance(msg, AIMessage):
                            st.text(f"AI: {msg.content}")
                del st.session_state.store[session_id]
            else:
                response = with_message_history.invoke([HumanMessage(content=user_input)], config=config)
                st.write(f"AI: {response.content}")
        else:
            st.warning("Please enter a Session ID.")

    st.subheader(f"Current Session History for Session ID: {session_id if session_id else 'None'}")
    if session_id in st.session_state.store:
        current_session_messages = st.session_state.store[session_id].messages
        for msg in current_session_messages:
            if isinstance(msg, HumanMessage):
                st.text(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.text(f"AI: {msg.content}")
    else:
        st.text("No history for this session yet.")

# --- Main Streamlit App --- 
st.sidebar.title("Chatbot Options")

selected_chatbot = st.sidebar.selectbox(
    "Choose a Chatbot:",
    (
        "Simple LLM",
        "Enhanced LLM",
        "Conversational Chatbot",
        "Session History with Conversation"
    )
)

if selected_chatbot == "Simple LLM":
    simple_llm_chatbot()
elif selected_chatbot == "Enhanced LLM":
    enhanced_llm_chatbot()
elif selected_chatbot == "Conversational Chatbot":
    conversation_chatbot()
elif selected_chatbot == "Session History with Conversation":
    session_history_with_conversation()