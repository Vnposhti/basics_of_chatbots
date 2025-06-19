from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
class State(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile")

from langgraph.graph import StateGraph, START, END
graph=StateGraph(State)

def chatbot(state:State):
    return {"messages": llm.invoke(state.messages)}

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app=graph.compile()

while True:
    user_input = input("('exit' to quit) You: ")

    if user_input.lower() == 'exit':
        print("AI: Goodbye!")
        break

    print("--------------------------------INVOCATION OUTPUT----------------------")

    # Invocation
    result=app.invoke({"messages":user_input})
    for m in result['messages']:
        m.pretty_print()

    print("--------------------------------STREAMING OUTPUT--------------------------------")

    # Streaming
    response=app.stream({"messages":user_input}, stream_mode="values")
    for event in response:
        for msg in event.values():
            for m in msg:
                m.pretty_print()