from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

def token_counter(messages):
    return sum(len(msg.content.split()) for msg in messages)

total_tokens=token_counter(messages)
print(f"Total no. of tokens: ", total_tokens)

from langchain_core.messages import trim_messages
trimmer=trim_messages(
    max_tokens=15,
    strategy="last",
    token_counter=token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

trimmed_msg=trimmer.invoke(messages)

for m in trimmed_msg:
    m.pretty_print()