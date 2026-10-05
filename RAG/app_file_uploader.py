'''
基于streamlit框架完成WEB网页上传服务
'''

import streamlit as st

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
uploader_file = st.file_uploader(
    "请上传TXT文件",
    type = ['txt'],
    accept_multiple_files = False
)

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # get_value 获取内容 -> bytes -> decode('utf-8)
    text = uploader_file.getvalue().decode('utf-8')
    st.write(text)