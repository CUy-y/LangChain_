import os
from pathlib import Path
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy
from typing_extensions import TypedDict

# 加载模型配置
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    model = "qwen-max",
    temperature = 0.7,
    api_key = api_key,
    base_url = base_url
)

@tool
def get_weather(city: str) -> str:
    '''获取城市天气'''
    return f"{city}这里是晴天"

@tool
def get_math_add(x: int, y: int) -> int:
    '''获取两数的和'''
    return (x+y)

tools = [get_math_add, get_weather]
tool_nodes = ToolNode(tools)

'''
LangGraph 状态管理的核心机制:
因为创建图的时候是显式指明的要创建MessageState结构的图
所以助手节点的返回值需要时字典形式, 这样图才知道咋更新
'''

def assistant(state: MessagesState, config: RunnableConfig) -> dict:
    system_prompt = "你是一个助手模型，帮助获取天气"
    all_messages = [SystemMessage(system_prompt)] + state["messages"]
    '''再把可以工具绑定到大模型的接口'''
    model = llm.bind_tools(tools)
    '''返回值包装成元组形式'''
    return {"messages": [model.invoke(all_messages)]}

'''
这里创建的条件边获取最后一条message, 获取的就是之前助手模型的最后一条消息，
也就说AImessge, 然后在里面检查是否有tool_calls
'''
def should_continue(state: MessagesState, config: RunnableConfig) -> str:
    messages = state["messages"]
    last_messgae = messages[-1]
    if last_messgae.tool_calls:
        return "continue"
    return "end"

# 创建图
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("assistant", assistant)
builder.add_node("tool", tool_nodes)

#添加边
builder.add_edge(START, "assistant")

# 添加条件边
builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)

# 添加边, 调用工具节点之后回到assistant
builder.add_edge("tool", "assistant")

# 编译图
mygraph = builder.compile(name = "my_graph")
# print(mygraph)

response = mygraph.invoke(
    {"messages": [HumanMessage(content = "威海天气咋样")]},
    version = "v2"
).value

for message in response["messages"]:
    message.pretty_print()