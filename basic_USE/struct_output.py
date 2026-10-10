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

structured_agent = create_agent(
    model = llm,
    tools = [wreather_tool, math_add],
    system_prompt = "你是我儿子",
    response_format = CalcInfo
)

response = structured_agent.invoke(
    {"messages": [{"role": "user","content": "计算 56565+8784"}]},
    config = {"configurable": {"thread_id": "1"}},
    context = Context(authority = "admin"),
    version = "v2"
).value

for message in response["messages"]:
    message.pretty_print()

print(response["structured_response"].output)