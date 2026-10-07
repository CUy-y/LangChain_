
import os
from pathlib import Path

from dotenv import load_dotenv


# 始终读取 RAG/.env，不受启动目录影响；文件中的值覆盖旧终端会话的配置。
load_dotenv(Path(__file__).with_name(".env"), override=True)
base_directory = Path(__file__).resolve().parent


def get_dashscope_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY，请在 RAG/.env 中配置百炼官方密钥后重新启动程序")
    return api_key


# Chroma
collection_name="rag"
persist_directory = str(base_directory / "chroma_db")
chat_history_directory = str(base_directory / "chat_history")

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"
dashscope_base_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
