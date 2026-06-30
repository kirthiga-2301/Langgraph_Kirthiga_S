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
"""You are Weather & Time agent. Your ONLY job is to help users with weather forecasts, current conditions, and time zone information.
CRITICAL RULES:
1. ONLY predict weather and time.
2. If the user asks ANY question unrelated to weather or time, you MUST NOT answer it. Instead, you MUST reply EXACTLY with: "I don't know. I can only help with weather and time."
3. Always use the appropriate tool whenever possible to get accurate data.
4. Never guess or hallucinate weather or time data if a tool exists.
5. Always explain the tool result in simple English (e.g., recommend an umbrella if it's raining)."""
)

def assistant(state):
    messages=[system]+state["messages"]

    last = state["messages"][-1]
    active_llm = llm_no_tools if isinstance(last, ToolMessage) else llm

    response = active_llm.invoke(messages)

    return {
        "messages":[response]
    }

def tools_node(state):
    last=state["messages"][-1]
    output=[]

    for call in last.tool_calls:
        name=call["name"]
        args=call["args"]

        result=tool_map[name].invoke(args)

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
graph.add_node("assistant",assistant)
graph.add_node("tools",tools_node)
graph.add_edge(START,"assistant")
graph.add_conditional_edges("assistant",router,{ "tools":"tools", END:END })
graph.add_edge("tools","assistant")
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
    print("\nAgent:",result["messages"][-1].content)
