from pathlib import Path
from uuid import uuid4

import streamlit as st

import config_data as config
from knowledge_base import KnowledgeBaseService
from rag import RagService


def decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030")


def safe_error(exc: Exception) -> str:
    return str(exc).replace(config.get_dashscope_api_key(), "[REDACTED]")[:400]


st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid4().hex
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()
if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = KnowledgeBaseService(
        vector_store=st.session_state["rag"].vector_service.vector_store
    )

knowledge_base = st.session_state["knowledge_base"]

with st.sidebar:
    st.header("参考资料")
    st.caption("上传 TXT 文件，或导入项目 data 目录中的 TXT 文件。")

    uploaded_files = st.file_uploader("选择 TXT 文件", type="txt", accept_multiple_files=True)
    if st.button("导入上传的文件", disabled=not uploaded_files):
        with st.spinner("正在导入资料..."):
            for uploaded_file in uploaded_files:
                try:
                    result = knowledge_base.upload_by_str(
                        decode_text(uploaded_file.getvalue()), uploaded_file.name
                    )
                    st.write(f"{uploaded_file.name}：{result}")
                except Exception as exc:
                    st.error(f"{uploaded_file.name} 导入失败：{type(exc).__name__}: {safe_error(exc)}")

    sample_files = sorted((Path(config.base_directory) / "data").glob("*.txt"))
    if st.button("导入 data 目录", disabled=not sample_files):
        with st.spinner("正在导入资料..."):
            for path in sample_files:
                try:
                    result = knowledge_base.upload_by_str(decode_text(path.read_bytes()), path.name)
                    st.write(f"{path.name}：{result}")
                except Exception as exc:
                    st.error(f"{path.name} 导入失败：{type(exc).__name__}: {safe_error(exc)}")

    chunk_count = knowledge_base.chroma._collection.count()
    st.caption(f"知识库当前有 {chunk_count} 个文本片段。")
    if chunk_count == 0:
        st.info("知识库为空，请先导入参考资料。")

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入问题")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    chunks = []

    def capture(generator):
        for chunk in generator:
            chunks.append(chunk)
            yield chunk

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并生成回答..."):
            try:
                session_config = {"configurable": {"session_id": st.session_state["session_id"]}}
                stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)
                st.write_stream(capture(stream))
                st.session_state["message"].append(
                    {"role": "assistant", "content": "".join(chunks)}
                )
            except Exception as exc:
                st.error(f"回答失败：{type(exc).__name__}: {safe_error(exc)}")
