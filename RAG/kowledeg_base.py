"""
知识库
"""

import os
import congfig_data as config
import hashlib


def check_md5(md5_str):
    """检查传入的md5字符串是否已经被处理"""

    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False

    with open(config.md5_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()

            if line == md5_str:
                return True

    return False


def save_md5(md5_str):
    """将传入的MD5字符串记录到文件内保存"""

    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str):
    """将传入的字符串转换为MD5字符串"""

    # 字符串 -> bytes
    str_bytes = input_str.encode("utf-8")

    # 创建MD5对象
    md5_obj = hashlib.md5()

    # 计算MD5
    md5_obj.update(str_bytes)

    # 获取十六进制字符串
    md5_hex = md5_obj.hexdigest()

    return md5_hex


class KnowledgeBaseService(object):

    def __init__(self):
        self.chroma = None
        self.spliter = None

    def upload_by_str(self, data, filename):
        pass


if __name__ == "__main__":
    # r1 = get_string_md5("周杰伦")
    # r2 = get_string_md5("周杰伦")
    # r3 = get_string_md5("周杰伦2")

    # print(r1)
    # print(r2)
    # print(r3)
    print(check_md5("7a8941058aaf4df5147042ce104568da"))