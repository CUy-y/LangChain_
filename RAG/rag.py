from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI




class RagService(object):
    def __init__(self):
        api_key = config.get_dashscope_api_key()

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name, dashscope_api_key=api_key, max_retries=2)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "请只根据参考资料回答用户问题，简洁、准确地说明结论，并标出来源文件名。"
                 "如果资料不足，请明确说资料中没有相关信息，不要编造。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=api_key,
            base_url=config.dashscope_base_url,
            timeout=30,
            max_retries=1,
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                source = doc.metadata.get("source", "未知文件")
                formatted_str += f"来源：{source}\n内容：{doc.page_content}\n\n"

            return formatted_str

        def format_for_retriever(value: dict)->str:

            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }| RunnableLambda(format_for_prompt_template) |self.prompt_template |self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)
