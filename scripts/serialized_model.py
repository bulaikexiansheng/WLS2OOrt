import pickle
import importlib.util as imu
import sys


def load_model_from_file(file_path):
    # 动态导入模型定义的模块
    spec = imu.spec_from_file_location("models", file_path)
    model_module = imu.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # 获取模型实例
    model = model_module.get_model_instance()

    return model


def serialize_and_save_model(model, output_file):
    # 将模型序列化为字节流
    serialized_model = pickle.dumps(model)

    # 保存序列化后的模型到文件
    with open(output_file, 'wb') as file:
        file.write(serialized_model)

    print(f"Serialized model saved to {output_file}")


if __name__ == "__main__":
    # 从命令行参数获取模型代码文件路径
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py path/to/model_file.py")
    #     sys.exit(1)
    #
    # model_file_path = sys.argv[1]
    model_file_path = "D:\研究生课程\MyOOrt\models\BlackNet.py"
    # 加载模型并序列化
    # try:
    loaded_model = load_model_from_file(model_file_path)
    serialize_and_save_model(loaded_model, "../models/serialized")
    print("模型序列化成功")
    # except Exception as e:
    #     print(f"Error: {e}")
    #     sys.exit(1)
