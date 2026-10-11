# LangChain / LangGraph Agent API 速查笔记

> **适用范围**：以 LangChain 1.x、LangGraph 1.1+ 的 Python API 为主，特别是 `create_agent()` 和 `version="v2"`。  
> **目标**：快速查阅 Pydantic 数据约束、工具注册、Runtime Context、中间件、Agent 调用、结果提取与三种流式输出。  
> **不包含**：`def`、普通 `class`、`self`、基本 Python 类型注解等语法入门。

## 目录

1. [Field：字段定义与验证](#1-field字段定义与验证)
2. [Literal：枚举式取值约束](#2-literal枚举式取值约束)
3. [@tool：注册 Agent 工具](#3-tool注册-agent-工具)
4. [Runtime Context：传递运行时数据](#4-runtime-context传递运行时数据)
5. [Callable：表示可调用接口](#5-callable表示可调用接口)
6. [@wrap_model_call：模型调用中间件](#6-wrap_model_call模型调用中间件)
7. [request.override()：修改当前模型请求](#7-requestoverride修改当前模型请求)
8. [create_agent()：组装 Agent](#8-create_agent组装-agent)
9. [.invoke()：同步调用 Agent](#9-invoke同步调用-agent)
10. [获取 Agent 的各种结果](#10-获取-agent-的各种结果)
11. [结构化输出 response_format](#11-结构化输出-response_format)
12. [同步流式输出 stream()](#12-同步流式输出-stream)
13. [updates 模式](#13-updates-模式)
14. [messages 模式](#14-messages-模式)
15. [values 模式](#15-values-模式)
16. [可复用的 Agent 完整模板](#16-可复用的-agent-完整模板)
17. [常见问题和排错](#17-常见问题和排错)
18. [官方参考资料](#18-官方参考资料)

---

## 1. `Field`：字段定义与验证

**来源**：`from pydantic import BaseModel, Field`

`Field()` 用于给 Pydantic 模型字段声明默认值、描述、范围等元数据和验证条件；常用于 **Agent 的输入/输出 Schema**、FastAPI 请求模型和工具参数模型。

```python
from pydantic import BaseModel, Field

class CalcInfo(BaseModel):
    """计算结果。"""
    output: int = Field(description="整数计算结果")
    explanation: str = Field(default="", description="过程说明")
```

**通用写法**：

```python
字段名: 字段类型 = Field(
    default=默认值,          # 可省略；省略且无 default_factory 时通常表示必填
    description="字段用途",  # 帮助开发者、模型理解字段语义
    ge=最小值,               # >=，主要用于数字
    le=最大值,               # <=，主要用于数字
)
```

| 参数 | 含义 | 示例 |
|---|---|---|
| `description` | 字段语义说明 | `Field(description="计算结果")` |
| `default` | 缺省时采用的默认值 | `Field(default=0)` |
| `default_factory` | 用函数生成默认值 | `Field(default_factory=list)` |
| `ge` / `gt` | 大于等于 / 大于 | `Field(ge=0)` |
| `le` / `lt` | 小于等于 / 小于 | `Field(le=100)` |
| `min_length` / `max_length` | 字符串、集合等长度限制 | `Field(min_length=1)` |

**使用与提取**：

```python
result = CalcInfo(output=65349)
print(result.output)       # 65349
print(result.model_dump()) # {'output': 65349, 'explanation': ''}
```

**易错点**：

- `Field(description=...)` **没有**设置默认值；`output` 仍是必填字段。
- `str | None` 表示 **允许 `None`**，但不等于允许省略；若允许省略应写 `name: str | None = None`。
- Pydantic 默认可能做类型转换，如把 `"123"` 转成 `123`。需要严格验证时，使用相应严格类型或 `strict=True`。
- 字段验证负责 **格式正确**，不负责验证计算结论是否符合事实。

## 2. `Literal`：枚举式取值约束

**来源**：`from typing import Literal`

`Literal` 用于限定某个字段只能取特定的字面值，适用于身份角色、任务类型、回答标签、运行模式等。

```python
from typing import Literal
from pydantic import BaseModel, Field

class Context(BaseModel):
    authority: Literal["admin", "user"] = "user"
    mode: Literal["fast", "accurate"] = Field(default="fast")

Context(authority="admin")   # 合法
Context(authority="guest")   # Pydantic ValidationError
```

**通用写法**：

```python
字段名: Literal["允许值1", "允许值2"]
```

> `Literal` 本身是类型注解；运行时检查由 Pydantic 等验证框架执行。它不会凭空实现业务权限控制。

## 3. `@tool`：注册 Agent 工具

**来源**：`from langchain.tools import tool`

`@tool` 将普通 Python 函数包装为 LangChain 的工具对象，便于向 LLM 提供 **工具名称、功能描述、输入参数 Schema**，并允许 Agent 自动调度调用。

```python
from langchain.tools import tool

@tool
def math_add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b
```

**常用操作**：

```python
print(math_add.name)                # math_add
print(math_add.description)         # 工具描述
print(math_add.args_schema)         # 工具参数 Schema
print(math_add.invoke({"a": 3, "b": 5}))  # 8
```

也可以显式指定工具名：

```python
@tool("sum_two_numbers")
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b
```

**关键理解**：

- 大模型决定是否调用、调用哪个工具以及传入哪些参数；LangChain 负责在 Python 环境中执行工具，然后把结果作为 `ToolMessage` 返回执行流程。
- `@tool` 的 **docstring 非常重要**，要写清什么时候调用及参数含义；没有描述时可能无法正常构建工具。
- 工具包装后，手动测试优先采用 `tool.invoke({"参数": 值})`，而不是直接当普通函数调用。
- `create_agent(tools=[...])` 在当前版本也支持某些普通 Python 函数；显式 `@tool` 有利于复用与自定义。
- `@tool` 不保证模型一定调用该工具；它只是赋予模型可调用能力。

## 4. Runtime Context：传递运行时数据

**目的**：提供一次 Agent 运行所需的 **非对话信息**，例如用户身份、租户 ID、权限配置。该信息主要由中间件、工具、图节点读取，**不会仅因传入 `context` 就自动写入模型提示词**。

### 4.1 定义 Context Schema

```python
from pydantic import BaseModel
from typing import Literal

class Context(BaseModel):
    authority: Literal["admin", "user"]
    user_id: str
```

### 4.2 创建 Agent 时声明 Schema

```python
agent = create_agent(
    model=llm,
    tools=[math_add],
    context_schema=Context,
)
```

### 4.3 调用时提供具体 Context 对象

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "计算 3+5"}]},
    context=Context(authority="admin", user_id="u001"),
)
```

- `context_schema=Context`：指定 **上下文的数据类型**。
- `context=Context(...)`：传入 **本次运行的具体值**。

### 4.4 工具中通过 `ToolRuntime` 读取 Context

```python
from langchain.tools import tool, ToolRuntime

@tool
def who_am_i(runtime: ToolRuntime[Context]) -> str:
    """读取当前用户信息。"""
    return f"用户={runtime.context.user_id}，身份={runtime.context.authority}"
```

`ToolRuntime[Context]` 由 LangChain 在工具执行阶段**注入**，不是要求 LLM 生成的业务参数。

`ToolRuntime` 还可用于访问执行状态、Store、流式写入器等，具体可用属性取决于配置与版本。

**安全边界**：

> `Context(authority="admin")` **不等于认证完成**。后端应根据已验证的登录身份确定此值。对于敏感工具，执行层仍要校验权限，不能只靠向模型隐藏工具名称。

## 5. `Callable`：表示可调用接口

**来源**：`from collections.abc import Callable`

通用形式：

```python
Callable[[参数类型1, 参数类型2], 返回值类型]
```

例子：

```python
from collections.abc import Callable

def execute(fn: Callable[[int, int], int]) -> int:
    return fn(3, 5)
```

在模型中间件的签名中：

```python
handler: Callable[[ModelRequest], ModelResponse]
```

含义是：`handler` 接收一个 `ModelRequest`，返回一个 `ModelResponse`。这里的 `handler` 是一个可调用对象，不是普通数据。

## 6. `@wrap_model_call`：模型调用中间件

**来源**：

```python
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from collections.abc import Callable
```

`@wrap_model_call` 允许在 **一次模型调用前/后** 插入逻辑，例如：权限过滤、动态模型选择、请求修改、日志、重试策略等。

**同步通用模板**：

```python
@wrap_model_call
def my_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 模型调用前
    print("即将发起模型请求")

    response = handler(request)  # 继续执行下一层或实际模型调用

    # 模型调用后
    print("模型请求已完成")
    return response
```

**核心参数**：

| 名称 | 作用 |
|---|---|
| `request` | 本次模型请求，包括消息、可用工具、runtime 等 |
| `handler` | 将请求交给下一层处理器执行的可调用对象 |
| `handler(request)` | 真正继续执行模型调用流程 |
| `response` | 模型调用得到的响应 |

执行次序：

```text
Agent 准备请求
      ↓
中间件前置操作
      ↓
handler(request)
      ↓
中间件后置操作
      ↓
return response
```

**常见错误**：

- 只写 `return request`：返回值类型不对；中间件通常应返回 `ModelResponse`。
- 忘记调用 `handler(request)`：模型请求无法按预期继续。
- 在同步处理器里随意使用 `await handler(...)`：同步与异步中间件签名不能混用。
- 这个装饰器针对 **模型调用**；不要将其和工具执行时的 `ToolRuntime` 混为一谈。

## 7. `request.override()`：修改当前模型请求

`request.override(...)` 会创建一份 **带有指定改动的当前模型请求**。常用于中间件中动态改变 `tools`、`system_prompt` 等模型调用相关配置（支持的字段以安装版本为准）。

典型场景：普通用户无法看到管理工具的 Schema。

```python
@wrap_model_call
def filter_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.authority != "admin":
        visible_tools = [
            t for t in request.tools
            if t.name != "math_add"
        ]
        request = request.override(tools=visible_tools)

    return handler(request)
```

**拆解**：

```python
request.runtime.context.authority  # 从运行时上下文获取身份
request.tools                      # 当前模型请求暴露的工具列表
t.name                             # 工具名称
request.override(tools=...)        # 生成修改后的模型请求
handler(request)                   # 使用修改后的请求继续执行
```

**与 `@tool` 的区别**：

- `@tool`：**注册工具**，赋予 Agent 调用该工具的能力。
- `request.override(tools=...)`：只修改 **当前请求提供给模型的工具列表**，不删除工具实现，也不等于独立的安全沙箱。

> 生产环境：权限控制应同时在工具执行层或业务服务端校验，避免其他执行路径绕过模型侧过滤。

## 8. `create_agent()`：组装 Agent

**来源**：`from langchain.agents import create_agent`

**通用模板**：

```python
agent = create_agent(
    model=llm,
    tools=[math_add, weather_tool],
    system_prompt="你是一个可使用工具的助手。",
    middleware=[filter_tools],
    context_schema=Context,
    response_format=CalcInfo,
    # checkpointer=...  # 可选，启用线程状态持久化
)
```

| 参数 | 职责 | 备注 |
|---|---|---|
| `model` | 模型/模型实例 | 例如 `ChatOpenAI(...)` |
| `tools` | Agent 可调度的工具集合 | 可为空列表 |
| `system_prompt` | 系统指令 | 不等于权限控制 |
| `middleware` | 传入中间件列表 | 运行时拦截模型调用 |
| `context_schema` | 运行时上下文数据结构 | 与 `invoke(context=...)` 对应 |
| `response_format` | 最终结构化输出 Schema 或策略 | 与最后一条普通消息不同 |
| `checkpointer` | 状态持久化实现 | 让 `thread_id` 真正关联历史状态 |

**注意**：`create_agent()` 组织了一套模型、工具及执行循环；最终是否真的调用某个工具由模型和执行配置决定。

## 9. `.invoke()`：同步调用 Agent

**通用写法**：

```python
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "计算 56565+8784"}
        ]
    },
    context=Context(authority="admin", user_id="u001"),
    config={"configurable": {"thread_id": "thread-001"}},
    version="v2",
)

state = result.value
```

| 参数/表达式 | 作用 |
|---|---|
| `messages` | 本次传给 Agent 的对话消息 |
| `context=...` | 传给运行时、工具、中间件的额外上下文 |
| `config=...` | 运行配置，如 `thread_id` |
| `version="v2"` | 启用 LangGraph v2 返回格式（需要 LangGraph 1.1+） |
| `result.value` | 取得 `GraphOutput` 中的 Agent 最终 State |
| `result.interrupts` | 查询发生的中断，通常为一个元组 |

**v1 与 v2 的区别**：

```python
# v1：返回 State（一般是 dict），通常不带 .value
state = agent.invoke(inputs)

# v2：返回 GraphOutput，再用 .value 获取 State
out = agent.invoke(inputs, version="v2")
state = out.value
```

**关于 `thread_id`**：

- 只有 `thread_id`，**没有**配置 `checkpointer`：不会自动生成持久化的多轮记忆。
- 配置了 Checkpointer 后，同一 `thread_id` 可以关联对应线程的状态。
- `InMemorySaver` 适合本地演示；进程重启即丢失。真实服务需要持久化 Checkpointer。

## 10. 获取 Agent 的各种结果

以下代码假定已经得到：

```python
state = agent.invoke(inputs, version="v2").value
```

### 10.1 最后一条消息的正文

```python
last_message = state["messages"][-1]
print(last_message.content)
```

`content` 可能是字符串或内容块列表，**不一定永远是文本字符串**。

### 10.2 打印所有消息，查看工具调用链

```python
for message in state["messages"]:
    message.pretty_print()
```

常见消息类型：

| 类型 | 代表什么 |
|---|---|
| `HumanMessage` | 用户输入 |
| `AIMessage` | 模型消息，可能包含工具调用意图 |
| `ToolMessage` | 实际工具执行结果 |
| `SystemMessage` | 系统指令（未必作为 State 列表的一项出现） |

### 10.3 提取工具调用记录

```python
for msg in state["messages"]:
    for call in getattr(msg, "tool_calls", []):
        print("工具名：", call["name"])
        print("参数：", call["args"])
        print("调用ID：", call.get("id"))
```

`AIMessage.tool_calls` 保存的是 **模型请求调用工具的记录**；工具执行后返回的内容通常在后续 `ToolMessage` 中。

### 10.4 提取结构化响应

```python
structured = state.get("structured_response")
if structured is not None:
    print(structured.output)   # 针对 Pydantic CalcInfo
    print(structured.model_dump())
else:
    print("本次没有获得结构化结果")
```

不要在未检查前就使用 `state["structured_response"].output`：如果值为 `None`，会发生 `AttributeError: 'NoneType' object has no attribute 'output'`。

## 11. 结构化输出 `response_format`

**作用**：让 Agent 在任务结束时生成符合指定 Schema 的数据，而不是让应用自己从自然语言里提取字段。

### 11.1 定义 Pydantic 输出 Schema

```python
from pydantic import BaseModel, Field

class CalcInfo(BaseModel):
    """计算结果。"""
    output: int = Field(description="计算结果")
```

### 11.2 自动选择策略

```python
agent = create_agent(
    model=llm,
    tools=[math_add],
    response_format=CalcInfo,
)
```

对于提供的 Schema 类，LangChain 会结合模型能力选择合适策略：

| 方式 | 写法 | 工作原理 |
|---|---|---|
| 自动 | `response_format=CalcInfo` | 依据模型特性自动选策略 |
| 工具策略 | `response_format=ToolStrategy(CalcInfo)` | 让模型通过结构化输出工具调用提交字段 |
| 供应商原生 | `response_format=ProviderStrategy(CalcInfo)` | 由支持 JSON Schema 等特性的模型 API 约束输出 |

```python
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

response_format = ToolStrategy(CalcInfo)       # 模型需要支持 Tool Calling
# response_format = ProviderStrategy(CalcInfo)  # 模型/API 需支持原生结构化输出
```

**特别注意**：

1. `response_format=CalcInfo` **不等于**“自动把最后一条自然语言回复解析为 CalcInfo”。
2. 如果同时使用其他工具和结构化输出，模型/API 需要具备相应的联合调用能力。
3. `structured_response` 可能是 `None`，或者不兼容的模型直接触发错误。不能把配置 Schema 当成成功保证。
4. 即便满足类型，Pydantic 也不会检查数学结果是不是正确。
5. 对于你使用的 **Qwen-Max / 阿里云百炼 OpenAI 兼容接口**，需要实测当前模型是否支持对应结构化输出策略；不能假设所有 Qwen 模型表现一致。

### 11.3 调用并检查返回

```python
state = agent.invoke(
    {"messages": [{"role": "user", "content": "计算 3+5"}]},
    version="v2",
).value

structured = state.get("structured_response")
if isinstance(structured, CalcInfo):
    print(structured.output)
else:
    print("结构化输出未生成或类型不符合预期")
    for msg in state["messages"]:
        msg.pretty_print()
```

---

## 12. 同步流式输出 `.stream()`

`agent.invoke()` 是等待执行完成后拿到整体结果；`agent.stream()` 让程序随着 Agent 运行持续接收事件。

**基本语法**：

```python
for part in agent.stream(
    {"messages": [{"role": "user", "content": "计算 3+5"}]},
    stream_mode="updates",
    version="v2",
):
    print(part)
```

**v2 流事件统一结构**（LangGraph 1.1+）：

```python
{
    "type": "updates",  # 当前事件所属的流模式
    "ns": (),          # 命名空间；子图场景可不为空
    "data": {...},     # 主要数据，具体形式因模式不同而不同
}
```

| 模式 | 返回什么 | 典型用途 |
|---|---|---|
| `"updates"` | **每个节点完成时的状态更新** | 看 Agent 怎么调用工具、调试执行流程 |
| `"messages"` | **模型输出的消息块及元信息** | 做逐字/逐块实时回答界面 |
| `"values"` | **图执行过程的完整状态快照** | 看某时刻有哪些消息、状态累积成什么 |

**不使用 `version="v2"` 时，流式输出的形状可能不同；下面的所有示例统一使用 v2，不能直接套用于 v1。**

## 13. `updates` 模式

**含义**：每当 Agent 的某个节点完成一轮执行，就返回该节点本次更新的数据（delta），**不是全文 token**，也不是完整 State。

```python
for part in agent.stream(
    {"messages": [{"role": "user", "content": "计算 3+5"}]},
    stream_mode="updates",
    version="v2",
):
    for node_name, update in part["data"].items():
        print("节点：", node_name)
        print("本次更新：", update)
```

常见节点（名称以实际构图为准）：

- `model`：模型生成的消息/工具调用请求。
- `tools`：工具执行产生的消息。

如果更新中有 `messages`：

```python
for part in agent.stream(inputs, stream_mode="updates", version="v2"):
    for node_name, update in part["data"].items():
        print("节点：", node_name)
        for msg in update.get("messages", []):
            msg.pretty_print()
```

**什么时候选**：想调试 Agent 的 ReAct 流程，看何时调用模型和工具时，优先选择此模式。

## 14. `messages` 模式

**含义**：逐块接收模型输出，可用于构建聊天界面的打字机效果，同时也可能收到工具调用相关的消息块。

```python
for part in agent.stream(
    {"messages": [{"role": "user", "content": "解释一下 LangGraph"}]},
    stream_mode="messages",
    version="v2",
):
    token, metadata = part["data"]

    # 简单的纯文本模型中通常直接是 str
    if isinstance(token.content, str) and token.content:
        print(token.content, end="", flush=True)
```

其中：

```python
token, metadata = part["data"]
```

- `token`：通常是 `AIMessageChunk`，表示本次模型生成的消息片段。
- `metadata`：包含来源信息等，例如 `metadata.get("langgraph_node")`。
- `token.content`：**可能是字符串，也可能是内容块列表**。仅处理字符串的示例不会完整展示多模态/复杂内容块。

若要针对文本块处理，可检查：

```python
for block in token.content_blocks:
    if block.get("type") == "text":
        print(block["text"], end="", flush=True)
```

> `.content_blocks` 可包含 reasoning、tool-call 等其他块类型。展示工具调用参数时，应当解析相应类型，而不是把所有内容一股脑拼成字符串。

**什么时候选**：FastAPI WebSocket / SSE 逐步展示模型回复；需要跟踪流式工具调用参数时也会涉及它。

## 15. `values` 模式

**含义**：持续获取整个 State 在不同阶段的快照，而不是只查看本次更改的字段。

```python
for part in agent.stream(
    {"messages": [{"role": "user", "content": "计算 3+5"}]},
    stream_mode="values",
    version="v2",
):
    state = part["data"]
    print("当前消息数量：", len(state["messages"]))
    print("当前最后一条消息：", state["messages"][-1].content)
```

同时查看可能存在的工具调用：

```python
for part in agent.stream(inputs, stream_mode="values", version="v2"):
    state = part["data"]
    last = state["messages"][-1]
    for call in getattr(last, "tool_calls", []):
        print(call["name"], call["args"])
```

`values` 相比 `updates` 含有更多累积状态，长对话时数据量通常更大。

### 15.1 同时订阅多个模式

```python
for part in agent.stream(
    inputs,
    stream_mode=["updates", "messages", "values"],
    version="v2",
):
    mode = part["type"]
    data = part["data"]

    if mode == "updates":
        print("节点变化：", data)
    elif mode == "messages":
        token, metadata = data
        if isinstance(token.content, str):
            print(token.content, end="", flush=True)
    elif mode == "values":
        print("当前状态中的消息数：", len(data["messages"]))
```

使用 v2 时，**不论订阅单个还是多个模式，外层都是包含 `type`、`ns`、`data` 的统一结构**。

### 15.2 异步对应关系（仅做备查）

| 同步 | 异步 |
|---|---|
| `agent.invoke(...)` | `await agent.ainvoke(...)` |
| `for x in agent.stream(...)` | `async for x in agent.astream(...)` |

本文重点是同步写法，不展开 `asyncio`、SSE 等异步框架细节。

---

## 16. 可复用的 Agent 完整模板

涵盖以下能力：

- 阿里云百炼 OpenAI 兼容接口初始化（沿用你目前的 `.env` 习惯）；
- `Context` 结构、`ToolRuntime` 工具内权限验证；
- 两个工具（加法、**模拟**天气）；
- `@wrap_model_call` + `request.override()` 动态过滤工具；
- `CalcInfo` 结构化输出；
- v2 `.invoke()`、消息历史与结构化结果提取；
- 可选的 `updates` 同步流演示。

**安装依赖**：

```bash
pip install -U langchain langgraph langchain-openai pydantic python-dotenv
```

**`.env` 示例**：

```env
DASHSCOPE_API_KEY=你的真实APIKey
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max
```

> 使用百炼专属接入地址也可以，但必须填支持 OpenAI 兼容协议的完整 API Base URL。不要把真实 API Key 写到公开仓库。

**`agent_demo.py` 完整代码**：

```python
import os
from collections.abc import Callable
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


# 1. 模型配置
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)


# 2. 运行时上下文：不是用户随便声称的权限，应由认证后端提供
class Context(BaseModel):
    authority: Literal["admin", "user"] = "user"


# 3. 工具：加法工具在执行层再次检查权限
@tool
def math_add(a: int, b: int, runtime: ToolRuntime[Context]) -> int:
    """计算两个整数的和；仅供管理员使用。"""
    if runtime.context.authority != "admin":
        raise PermissionError("只有管理员可以执行 math_add")
    return a + b


@tool
def weather_mock(city: str) -> str:
    """返回某城市的模拟天气；不查询真实天气。"""
    return f"{city}的模拟天气是晴天"


# 4. 结构化输出数据模型
class CalcInfo(BaseModel):
    """整数计算结果"""
    output: int = Field(description="最终计算得到的整数")


# 5. 动态工具过滤中间件
@wrap_model_call
def filter_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.authority != "admin":
        request = request.override(
            tools=[t for t in request.tools if t.name != "math_add"]
        )
    return handler(request)


# 6. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[math_add, weather_mock],
    system_prompt=(
        "你是整数计算助手。遇到加法问题，优先使用 math_add 工具。"
        "完成计算后尝试按指定的 CalcInfo 结构返回结果。"
    ),
    middleware=[filter_tools],
    context_schema=Context,
    response_format=CalcInfo,
    checkpointer=InMemorySaver(),  # 本地测试用，进程退出后不持久化
)


# 7. 一次调用
inputs = {
    "messages": [
        {"role": "user", "content": "计算 56565+8784"}
    ]
}
config = {"configurable": {"thread_id": "demo-admin-001"}}
context = Context(authority="admin")

result = agent.invoke(
    inputs,
    config=config,
    context=context,
    version="v2",
)
state = result.value


# 8. 查看对话/工具执行链
print("\n===== 消息执行链 =====")
for msg in state["messages"]:
    msg.pretty_print()


# 9. 提取结构化结果：避免 NoneType 报错
print("\n===== 结构化结果 =====")
structured = state.get("structured_response")
if isinstance(structured, CalcInfo):
    print("计算答案：", structured.output)
    print("结果字典：", structured.model_dump())
else:
    print("本次没有获得 CalcInfo 结构化结果")
    print("最后一条消息：", state["messages"][-1].content)


# 10. 可选：再执行一次以观察 updates 事件（会再次调用模型并消耗额度）
# 注意：若使用相同 thread_id，检查点会延续之前的线程状态。
# 如果希望独立演示，使用另一个 thread_id。
# for part in agent.stream(
#     inputs,
#     config={"configurable": {"thread_id": "demo-stream-001"}},
#     context=context,
#     stream_mode="updates",
#     version="v2",
# ):
#     print("事件类型：", part["type"])
#     print("节点更新：", part["data"])
```

**怎样复用模板**：

1. 更换 `llm` 的 `model`、`api_key`、`base_url`，适配新的服务商。
2. 新增一个 `@tool` 函数，并加到 `tools=[...]`。
3. 如果有用户权限等环境信息，扩展 `Context`，在工具中使用 `ToolRuntime[Context]`。
4. 需要动态限制工具时修改 `filter_tools`；**敏感工具在执行层仍要校验**。
5. 任务输出不是一个整数时，修改 `CalcInfo`；如果只想聊天，移除 `response_format`。
6. 如果需要长期多轮记忆，把 `InMemorySaver` 换成持久化的 Checkpointer。

**模型兼容性提示**：示例代码在语法上展示这些组件如何协作，但是否成功生成 `CalcInfo` 仍取决于具体模型/API 对 Tool Calling 和结构化输出的支持。`qwen-max` 的实际表现以你当前百炼接口的运行结果为准。

---

## 17. 常见问题和排错

| 现象 | 主要可能原因 | 首先检查什么 |
|---|---|---|
| `structured_response` 是 `None` | 模型只回答普通文本，未走完结构化输出过程 | 打印每一条 `AIMessage` / `ToolMessage`；确认模型支持结构化输出策略 |
| `'NoneType' object has no attribute 'output'` | 在 `None` 上访问 `.output` | 改用 `state.get(...)` 并判断 `is not None` |
| 调用 `.value` 报错 | 旧版本或默认 v1 返回对象是字典 | 确认 LangGraph 版本与 `version="v2"` |
| 模型不调用 `math_add` | 工具使用由模型决策，提示词不等于强制调度 | 检查 docstring、工具是否注册、模型 Tool Calling 能力 |
| `thread_id` 不保留历史 | 未配置 `checkpointer` 或复用了错误线程 | 查看 `create_agent(checkpointer=...)` 和 `thread_id` |
| `@wrap_model_call` 中出错 | 参数签名错误、忘记 `handler(request)` | 对照同步中间件模板 |
| 普通用户仍可执行敏感操作 | 仅靠模型侧过滤，没有后端执行校验 | 在工具内部和业务接口校验已认证身份 |
| `messages` 模式输出一堆列表 | `token.content` 不是纯字符串 | 检查 `token.content_blocks`，仅显示合适的文本块 |
| 流式代码访问 `part["data"]` 报错 | 使用了 v1 流格式 | 升级到 LangGraph 1.1+ 并指定 `version="v2"` |
| 环境变量为 `None` | `.env` 没加载或变量名有误 | `load_dotenv()`、变量名称、路径、API 地址 |

**排错顺序建议**：

```text
先确认模型 API 能单独调用
  → 再验证一个 @tool 的执行
  → 再用 create_agent + invoke 看完整消息
  → 再加入 Context / Middleware
  → 最后测试 response_format 和 stream
```

这样更容易定位究竟是接口、工具、上下文、模型策略，还是返回结果解析的问题。

---

## 18. 官方参考资料

- 教程原文：[LangGraph 1.0 完全指南：快速入门](https://www.luochang.ink/dive-into-langgraph/quickstart/)
- LangChain：[Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- LangChain：[Streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- LangGraph：[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

> **版本提示**：LangChain / LangGraph 的 API 更新频繁；`version="v2"` 和本文的统一 `StreamPart` 写法要求 LangGraph 1.1 或更新版本。运行时请以安装环境的实际版本和官方参考文档为准。
