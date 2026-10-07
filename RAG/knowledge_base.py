"""
知识库
"""

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def get_string_md5(input_str:str ,encoding='utf-8'):
    """将出传入的字符串转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5 对象
    md5_obj =hashlib.md5()      # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容（传入即将要转换的字节数组）
    md5_hex=md5_obj.hexdigest() # 得到md5的十六进制字符串

    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self, vector_store=None):
        # 如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma = vector_store if vector_store is not None else Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(
                model=config.embedding_model_name,
                dashscope_api_key=config.get_dashscope_api_key(),
                max_retries=2,
            ),
            persist_directory=config.persist_directory,
        )
        self.spliter=RecursiveCharacterTextSplitter(  # 文本分割器的对象
            chunk_size=config.chunk_size,             # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,       # 连续文本段之间的字符重叠数量
            separators=config.separators,             # 自然段落划分的符号
            length_function=len,                      # 使用python自带的len函数做长度统计的依赖
        )      # 文本分割器的对象

    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        if not data.strip():
            return "[跳过] 文件内容为空"

        md5_hex=get_string_md5(data)
        if self.chroma.get(where={"content_md5": md5_hex}, limit=1, include=[])["ids"]:
            return "[跳过] 内容已存在知识库"
        if len(data) > config.max_spliter_char_number:
            knowledge_chunks:list[str]=self.spliter.split_text(data) # 类型统一，均用列表套字符串
        else:
            knowledge_chunks=[data]

        metadata={
            "source":filename,
            "content_md5":md5_hex,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"客户",
        }

        self.chroma.add_texts(        # 内容加载到向量库中
            # iterable-> list \tuple
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],

        )
        return "[Success]内容已经成功载入向量库"

if __name__ =='__main__':
    service= KnowledgeBaseService()
    r=service.upload_by_str("流星","testfile")
    print(r)
