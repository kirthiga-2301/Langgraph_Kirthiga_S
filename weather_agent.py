import os
import random
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langgraph.graph import (
    StateGraph,
    START,
    END
)
from typing import TypedDict, Annotated
import operator

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Please set GROQ_API_KEY in your .env file.")
    exit()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0
)

@tool
def get_weather(location: str):
    """Get the weather for a location"""
    cond = random.choice(["Sunny", "Cloudy", "Rainy"])
    temp = random.randint(10, 35)
    return f"The weather in {location} is {cond} and {temp}°C."

@tool
def get_time():
    """Get the current time"""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

tools=[get_weather, get_time]

tool_map={
    t.name:t
    for t in tools
}

llm_no_tools = llm          
llm = llm.bind_tools(tools) 

class State(TypedDict):
    messages: Annotated[list,operator.add]

system = SystemMessage(
"""
You are a helpful assistant.
Only answer weather and time questions.
Available tools:
get_weather
get_time
Use only one required tool.
After tool result, give final answer.
Do not call another tool.
"""
)

def assistant(state):
    messages=[system]+state["messages"]
    print("\n[LLM Request]")
    for m in messages:
        print(
            m.type,
            ":",
            m.content
        )

    last = state["messages"][-1]
    active_llm = llm_no_tools if isinstance(last, ToolMessage) else llm

    response = active_llm.invoke(messages)
    print("\n[LLM Response]")

    if response.tool_calls:
        print(
            "Tool:",
            response.tool_calls
        )
    else:
        print(
            response.content
        )

    return {
        "messages":[response]
    }

def tools_node(state):
    last=state["messages"][-1]
    output=[]
    print("\n[Tools]")

    for call in last.tool_calls:
        name=call["name"]
        args=call["args"]
        print(
            "Running:",
            name
        )

        result=tool_map[name].invoke(args)
        print(
            "Result:",
            result
        )

        output.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            )
        )

    return {
        "messages":output
    }

def router(state):
    last=state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

graph=StateGraph(State)
graph.add_node(
    "assistant",
    assistant
)
graph.add_node(
    "tools",
    tools_node
)
graph.add_edge(
    START,
    "assistant"
)
graph.add_conditional_edges(
    "assistant",
    router,
    {
        "tools":"tools",
        END:END
    }
)
graph.add_edge(
    "tools",
    "assistant"
)
agent=graph.compile()

print("\n=== LangGraph trace (Weather & Time) ===")

while True:
    user=input("\nYou: ")
    if user.lower()=="exit":
        print("Agent: Bye")
        break

    result = agent.invoke(
        {
            "messages":[
                HumanMessage(
                    content=user
                )
            ]
        }
    )
    print(
        "\nAgent:",
        result["messages"][-1].content
    )
