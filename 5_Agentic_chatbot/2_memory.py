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

from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()

from langgraph.graph import StateGraph, START, END
graph=StateGraph(State)

def chatbot(state:State):
    return {"messages": llm.invoke(state.messages)}

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app=graph.compile(checkpointer=memory)

while True:
    thread_id = input("('stop' to quit) Session ID: ")
    if thread_id.lower() == 'stop':
        break

    config={"configurable":{"thread_id":thread_id}}   
    
    while True:
        user_input = input("('exit' to quit) You: ")

        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            break

        # Invocation
        result=app.invoke({"messages":user_input},config)
        print(result['messages'][-1].content)

        print("="*135)

        # stream mode = "updates" --> streams only the updates to state of the graph after each node is called
        response=app.stream({"messages":user_input},config, stream_mode="updates")
        for event in response:
            for value in event.values():
                print(value["messages"].content)

        print("="*135)

        # stream mode = "values" --> streams the full state of the graph after each node is called
        response1=app.stream({"messages":user_input},config, stream_mode="values")
        for event in response1:
            for msg in event.values():
                for m in msg: # m = Human Message or AI Message
                    m.pretty_print()