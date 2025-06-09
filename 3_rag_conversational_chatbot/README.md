# 🧠 Conversational RAG Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **FastAPI**, **FAISS**, **HuggingFace Embeddings** and **Groq's LLaMA3 model**. It supports PDF-based document ingestion, semantic retrieval, and multi-turn chat history for context-aware Q&A.

---

## 📦 Features

* 🔍 **Semantic search** using **FAISS** and **HuggingFace embeddings**
* 📄 Supports **PDF ingestion** and text chunking
* 💬 Maintains **session-wise chat history**
* 🤖 Powered by **Groq's ultra-fast llama3-8b-8192 model**
* 🚀 **RESTful API** built with **FastAPI**
* 🧠 **Context-aware RAG** with historical memory

---

## 📤 How It Works

* **PDF Loading:** Loads and splits PDF into overlapping chunks.
* **Embeddings:** Converts chunks to vectors using all-MiniLM-L6-v2.
* **FAISS Vector Store:** Stores and retrieves relevant chunks based on semantic similarity.
* **RAG Chain:** Constructs responses by combining historical chat context and retrieved content from the PDF.
* **FastAPI Endpoint:** Handles incoming chat requests, managing session-wise memory to provide context-aware answers.

---

## 🛠️ Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/rejoicehub-interview-lab/559-Poshtiwala-Vishal-Nareshkumar-III.git
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables**

    Create a `.env` file in the root directory:

    ```env
    GROQ_API_KEY=your_groq_api_key
    ```

---

## 🚀 Run the API Server

```bash
uvicorn main:app --reload
```

The server will be available at: http://127.0.0.1:8000

**Go to `Swagger Docs:`** http://127.0.0.1:8000/docs

**Endpoint**
`POST /chat`

**Request Body**
{
  "session_id": "user123",
  "question": "What is the summary of this PDF?"
}

**Response Body**
{
  "session_id": "user123",
  "response": "Here is the answer based on the PDF content...",
  "chat_history": [
    {
      "type": "human",
      "content": "What is the summary of this PDF?"
    },
    {
      "type": "ai",
      "content": "Here is the answer based on the PDF content..."
    }
  ]
}

https://rag-chatbots.streamlit.app/