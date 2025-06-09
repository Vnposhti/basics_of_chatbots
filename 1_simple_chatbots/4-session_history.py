# Load virtual environment
from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")

from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(llm,get_session_history)

while True:
    session_id = input("('stop' to quit) Session ID: ")
    if session_id.lower() == 'stop':
        break

    config={"configurable":{"session_id":session_id}}   
    
    while True:
        user_input = input("('exit' to quit) You: ")

        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break

        response=with_message_history.invoke(
            [HumanMessage(content=user_input)],
            config=config
        )
        print(response.content)