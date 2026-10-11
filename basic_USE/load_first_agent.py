import os
from dotenv import load_dotenv
from IPython.display import HTML, display, update_display
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
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

# agent = create_agent(
#     model = llm,
#     system_prompt="你是我儿子"
# )

# output = agent.invoke({"messages": "你好"}, version = "v2")
# response = output.value

# print(response["messages"][-1].content)

# 创建一个带有工具调用的Agent

def get_weather(city: str) ->str:
    """获取当地的天气"""
    return f"{city}这里是晴天"

tool_agent = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt = "你是我儿子"
)

response = tool_agent.invoke({"messages": "今天济宁啥天气"}, version = "v2").value

print(response["messages"][-1].content)