import os
from pathlib import Path
from dotenv import load_dotenv
from collections.abc import Callable
from typing import Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool
from IPython.display import display, update_display, HTML

load_dotenv()
apikey = os.getenv("DASHSCOPE_API_KEY")
baseurl = os.getenv("DASHSCOPE_BASE_URL")

print("API Key 是否加载成功：", bool(apikey))
print(baseurl)

llm = ChatOpenAI(
    model = "qwen-max",
    api_key = apikey,
    base_url = baseurl
)


class Context(BaseModel):
    authority: Literal["admin", "user"]


@tool
def math_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def wreather_tool(city: str) -> str:
    '''获取当地天气'''
    return f"{city}这里的天气是晴天！"

class CalcInfo(BaseModel):
    '''计算信息'''
    output: int = Field(description = "计算结果")

# structured_agent = create_agent(
#     model = llm,
#     tools = [wreather_tool, math_add],
#     system_prompt = "你是我儿子",
#     response_format = CalcInfo
# )

# response = structured_agent.invoke(
#     {"messages": [{"role": "user","content": "计算 56565+8784"}]},
#     config = {"configurable": {"thread_id": "1"}},
#     context = Context(authority = "admin"),
#     version = "v2"
# ).value

# for message in response["messages"]:
#     message.pretty_print()

# print(response["structured_response"].output)

agent = create_agent(
    model = llm,
    tools = [wreather_tool]
)


# stream流式输出，使用for循环再agent工作中不断获取产生的数据
# 实际上模型的执行过程分为三部，首先是进入model层，然后是调用工具层tools
# ，最后model层是返回最终结果

# 核心标志：stream_mode = "updates"  # 代表流式输出模式
# 

# for part in agent.stream(
#     {"messages": [{"role": "user", "content": "济宁是什么天气"}]},
#     stream_mode = "updates",
#     version = "v2"
# ):
#     for step, data in part["data"].items():
#         print(f"当前节点: {step}")

#         msg = data["messages"][-1]

#         if step == "model":
#             if msg.tool_calls:
#                 print("模型请求调用工具：")
#                 for call in msg.tool_calls:
#                     print(f"工具名: {call["name"]}")
#                     print(f"参数: {call["args"]}")
#             else:
#                 print(f"模型最终回答: {msg.content}")
#         elif step == "tools":
#             print(f"工具调用结果: {msg.content}")

# messages模式

# pre_node = ""

# for part in agent.stream(
#     {"messages": [
#         {"role": "user", "content": "济宁今天是啥天气？"}
#     ]},
#     stream_mode = "messages",
#     version = "v2"
# ):
#     token, metadata = part["data"]
#     node = metadata["langgraph_node"]

#     if node != pre_node:
#         print(f"当前节点: {node}")
#         pre_node = node

#     if isinstance(token.content,str) and token.content:
#         print(token.content, end = "", flush = True)

# value模式，获得模型当前的完整状态，在langGraph中最有价值
for part in agent.stream(
    {"messages": [
        {"role": "user", "content": "济宁今天天气怎么样"}
    ]},
    stream_mode="values",
    version="v2"
):
    state = part["data"]
    print("当前完整状态:")
    print(state)
    
'''
对比三种模式的数据结构：

updata: 元组
part["data"] = {
    "model": {
        "messages": [...]
    }
}

messages: 字典
part["data"] = (
    AIMessageChunk(...),
    {"langgraph_node": "model"}
)

values: 元组
part["data"] = {
    "messages": [...]
}

'''