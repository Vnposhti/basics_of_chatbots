# Load virtual environment
from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama3-8b-8192")

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

conversation_history = []

while True:
    user_input = input("('exit' to quit) You: ")

    if user_input.lower() == 'exit':
        print("AI: Goodbye!")
        break

    conversation_history.append(HumanMessage(content=user_input))
    response = llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

print("\n--- Full Conversation History ---")
for msg in conversation_history:
    if isinstance(msg, HumanMessage):
        print(f"Human: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"AI: {msg.content}")