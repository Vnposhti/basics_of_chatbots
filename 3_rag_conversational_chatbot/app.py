import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
st.set_page_config(layout="wide")

uploaded_pdf=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)

@st.cache_resource
def setup_vectorstore(uploaded_pdf):
    """
    Processes the PDF content and creates a FAISS vector store.
    This function is cached and only runs if pdf_file_content changes.
    """
    try:
        if uploaded_pdf:
            documents=[]
            for pdf in uploaded_pdf:
                temppdf=f"./temp.pdf"
                with open(temppdf,"wb") as file:
                    file.write(pdf.getvalue())
                    file_name=pdf.name
                
                with st.spinner("Data is being loaded"):    
                    loader=PyPDFLoader(temppdf)
                    document=loader.load()
                    documents.extend(document)
            with st.spinner("Chunking in Progress"):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
            with st.spinner("Storing in vector database"):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)
            os.unlink(temppdf)
            return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- Chatbot Functions ---

# RAG QnA
def qna_chatbot():
    st.title("RAG Q&A Chatbot")

    llm = ChatGroq(model_name="llama3-8b-8192")
    vectorstore= setup_vectorstore(uploaded_pdf)
    
    if vectorstore is None:
        return   
    
    with st.spinner("Retrieval in Progress"):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_input = st.text_input("Ask questions based on the document:", key="qna_user_input")

    if st.button("Answer", key="qna_send_button") and user_input:
        with st.spinner("Getting answer..."):
            response = rag_chain.invoke({"input": user_input})
            st.write(response['answer'])

# 2-conversation_with_session_history.py functionality

if 'store' not in st.session_state:
    st.session_state.store={}

def get_session_history_func(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def conversational_rag_chatbot():
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.title("Conversational RAG Chatbot")
        session_id=st.text_input("Session ID",value="abc123")
        llm = ChatGroq(model_name="llama3-8b-8192")
        vectorstore= setup_vectorstore(uploaded_pdf)
        
        if vectorstore is None:
            return   
        
        with st.spinner("Retrieval in Progress"):
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        contextualize_q_system_prompt = (
            "You are an expert at extracting the core intent from a conversation. "
            "Given the entire chat history and the latest user message, "
            "your task is to formulate a clear, concise, and standalone question that can be used to search for relevant information. "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question." 
            "If you don't know the answer, say that you don't know."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask anything:", key="conv_rag_user_input")
        config = {"configurable": {"session_id": session_id}}

        if st.button("Send", key="conv_rag_send_button") and user_input:
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config=config
                )
                st.text(response['answer'])
            # st.rerun()

        if st.button("Clear Session History", key="conv_rag_clear_history"):
            if session_id in st.session_state.store:
                del st.session_state.store[session_id]
                st.success(f"Session history for ID '{session_id}' cleared.")
                st.rerun()

    # Display existing messages
    with col2:
        st.title("Chat History")
        if session_id in st.session_state.store:
            for msg in reversed(st.session_state.store[session_id].messages):
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

# --- Main Streamlit App --- 
st.sidebar.title("Chatbot Options")

selected_chatbot = st.sidebar.selectbox(
    "Choose a Chatbot:",
    (
        "RAG Q&A",
        "Conversational RAG"
    )
)

if selected_chatbot == "RAG Q&A":
    qna_chatbot()
elif selected_chatbot == "Conversational RAG":
    conversational_rag_chatbot()
