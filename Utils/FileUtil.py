
import os
def readFileToString(filePath):
    fileContent = ""
    with open(filePath, "rb") as file:
        while True:
            line = file.readline()
            if not line:
                break
            fileContent = "".join([fileContent, line.decode("utf-8")])
    return fileContent


def get_file_size(file_path):
    try:
        if not file_path:
            return 0
        # 获取文件大小（以字节为单位）
        size_in_bytes = os.stat(file_path).st_size
        return size_in_bytes
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None