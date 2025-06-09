import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load virtual environment
from dotenv import load_dotenv
load_dotenv()

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question):
    llm=ChatGroq(model_name="llama3-8b-8192")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

st.title("Simple Q&A Chatbot")
user_input=st.text_input("Ask me anything:","Tell me about LLMs")

if st.button ('Answer'):
    response=generate_response(user_input)
    st.write(response)