from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load virtual environment
from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")

# PDF Processing, chunking, embedding and vector storage
pdf_path = "sample.pdf"
loader = PyPDFLoader(pdf_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Rag Prompt
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

# Query Prompt
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

# Storage of session-wise chat history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Rag-chain with chat history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# FastAPI App
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(
    title="Conversational RAG Chatbot",
    description="A chatbot powered by LangChain, Groq, FAISS, and RAG with conversational history."
)

# Pydantic classes for chat request and response 
class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    chat_history: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles a chat interaction, leveraging RAG with conversational history.
    """
    session_id = request.session_id
    user_question = request.question
    config = {"configurable": {"session_id": session_id}}

    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config=config
    )

    current_history = get_session_history(session_id).messages
    formatted_history = [
        {"type": msg.type, "content": msg.content} for msg in current_history
    ]

    return ChatResponse(
        session_id=session_id,
        response=response["answer"],
        chat_history=formatted_history
    )