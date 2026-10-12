import os
import asyncio
import logging
from pathlib import Path
from typing import NotRequired, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy, TimeoutPolicy, Command
from langgraph.errors import NodeError, NodeTimeoutError

from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


# ========== 1. 模型配置 ==========

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

if not api_key or not base_url:
    raise RuntimeError("请检查 .env 中的 API Key 和 Base URL")

llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
    max_retries=0,  # 关闭模型客户端的内部重试
)


# ========== 2. 定义工具 ==========

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}这里是晴天"


@tool
def get_math_add(x: int, y: int) -> int:
    """计算两个数的和"""
    return x + y


tools = [get_math_add, get_weather]
tool_nodes = ToolNode(tools)

# 将工具绑定到模型
model = llm.bind_tools(tools)

# 先定义state
class AgentState(MessagesState):
    error_info: NotRequired[str]

# 改为异步助手节点
async def assistant(
        state: AgentState,
        config: RunnableConfig
) -> dict:

    system_prompt = "你是一个助手模型，帮助获取天气"

    all_messages = [
        SystemMessage(content = system_prompt)
    ] + state["messages"]

    response = await model.ainvoke(
        all_messages,
        config = config
    )

    return {"messages": [response]}

# 定义条件边
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "continue"

    return "end"

# 错误处理函数
def handle_assistant_error(
    state: AgentState,
    error: NodeError
) -> Command:

    error_info = (
        f"节点 {error.node} 执行失败："
        f"{type(error.error).__name__}: {error.error}"
    )

    # 记录详细故障，便于开发调试
    logging.error("%s", error_info)

    # 更新 State，并转入 fallback 节点
    return Command(
        update={"error_info": error_info},
        goto="fallback"
    )

# 失败回退节点
def fallback(state: AgentState) -> dict:
    return{
        "messages": [
            AIMessage(
                content = "模型暂时不可用, 请稍后再试"
            )
        ]
    }

# 创建图
builder = StateGraph(AgentState)

builder.add_node(
    "assistant",
    assistant,

    retry_policy = RetryPolicy(
        max_attempts = 3, # 最多尝试三次
        initial_interval = 1.0, # 初始重试等待间隔为1s
        backoff_factor = 2.0, # 重试间隔按照指数增长
        max_interval = 5.0, # 限制最大等待间隔
        jitter = True, # 给等待时间添加随机扰动，减少并发重试
        retry_on = ( # 指定哪些重述值得尝试
            ConnectionError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            NodeTimeoutError,
        ),
    ),

    timeout = TimeoutPolicy(
        run_timeout = 90,
        idle_timeout = 45,
    ),

    # 最终失败之后的处理函数
    error_handler = handle_assistant_error, 
)

builder.add_node("tool", tool_nodes)
builder.add_node("fallback", fallback)

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tool",
        "end": END,
    },
)

builder.add_edge("tool", "assistant")
builder.add_edge("fallback", END)

mygraph = builder.compile(name = "my_graph")

async def main():
    response = await mygraph.ainvoke({
        "messages": [
            HumanMessage(content = "威海天气咋样")
        ]
    })

    for message in response["messages"]:
        message.pretty_print()

    if "error_info" in response:
        print("错误记录: ",response["error_info"])

if __name__ == "__main__":
    asyncio.run(main())