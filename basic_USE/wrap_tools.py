import os
from pathlib import Path
from dotenv import load_dotenv
from collections.abc import Callable
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool
from langchain.agents import create_agent

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

@wrap_model_call
def filter_model_tool(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
):
    '''根据权限过滤工具调用'''
    if request.runtime.context.authority != "admin":
        request = request.override(
            tools = [tool for tool in request.tools if tool.name != "math_add"]
        )
    return handler(request)



tool_agent = create_agent(
    model = llm,
    tools = [wreather_tool,math_add],
    middleware = [filter_model_tool],
    context_schema = Context,
    system_prompt = "你是我儿子"
)

response = tool_agent.invoke(
    {"messages": [{"role": "admin", "user": "请计算 88888 + 55555 = "}]},
    context = Context(authority = "admin"), 
    version = "v2"
).value

print(response["messages"][-1].content)