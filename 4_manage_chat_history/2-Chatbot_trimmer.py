# Load virtual environment
from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")

from langchain_core.output_parsers import StrOutputParser
outputparser=StrOutputParser()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a sarcastic assistant."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def token_counter(messages):
    return sum(len(msg.content.split()) for msg in messages)

trimmer=trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

input_wrapper = RunnableLambda(lambda x: {"messages": x}) # To transform list input into dict for RunnablePassthrough.assign()
trimmed_msg = RunnablePassthrough.assign(messages=RunnableLambda(lambda x: trimmer.invoke(x["messages"])))
chain = (input_wrapper | trimmed_msg | prompt | llm | outputparser)
with_message_history=RunnableWithMessageHistory(chain,get_session_history)

while True:
    session_id = input("('stop' to quit) Session ID: ")
    if session_id.lower() == 'stop':
        break

    config={"configurable":{"session_id":session_id}}       

    while True:
        user_input = input("('exit' to quit) You: ")

        if user_input.lower() == 'exit':
            
            print(f"\n--- Full Conversation History for Session ID - {session_id} ---")
            conversation_history = store[session_id].messages
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    print(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"AI: {msg.content}")
            
            print("AI: Goodbye!")
            break

        response=with_message_history.invoke(
            {"messages":[HumanMessage(content=user_input)]},
            config=config
        )
        
        print(response)