import hashlib


def calculate_file_signature(file_path, hash_algorithm="sha256"):
    # 创建哈希对象
    hash_object = hashlib.new(hash_algorithm)

    # 以二进制模式打开文件
    with open(file_path, 'rb') as file:
        # 逐块更新哈希对象
        for chunk in iter(lambda: file.read(4096), b''):
            hash_object.update(chunk)

    # 返回十六进制表示的哈希值
    return hash_object.hexdigest()
