import os
import socket
import time
from pprint import pprint
import torch
from Utils.FileUtil import get_file_size
from Utils.DatasetTool import getDataset
from Connection.OOrtRequest import  connectToSever
import pickle
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from Utils.PrintTitle import showTitle
from Utils.XMLTool import parseXmlFileToDict


def load_cifar(batch_size=64, num_workers=4):
    """
    本地训练数据和测试数据
    :param batch_size:
    :param num_workers:
    :return:
    """
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_cifar)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=True)
    return train_loader, test_loader


def train_per_epoch(epoch, model, train_loader, optimizer, criterion, print_interval=10):
    loss_memory = 0.0
    times = 0

    for i, data in enumerate(train_loader, 0):
        times += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.view(-1, torch.prod(torch.tensor(inputs.size())[1:])))  # 将输入的维度调整为展平后的大小
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_memory += loss.item()

    # 在每个epoch结束时输出损失
    average_loss = (loss_memory / times) ** 2
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.3f}')

    return (loss_memory / times) ** 2


def train(model, train_loader, optimizer, criterion, epochs):
    """
    训练函数
    :param model:模型
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param epochs:
    :return:
    """
    showTitle("本地训练开始")
    start_time = time.time()
    loss_2_list = []
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        loss_2_list.append(train_per_epoch(epoch, model, train_loader, optimizer, criterion))
    end_time = time.time()
    elapsedTime = end_time - start_time
    showTitle("本地训练结束")
    return [elapsedTime, sum(loss_2_list)]


def loadModel(pickleFilePath, paramFilePath):
    """
    加载模型和参数
    :param pickleFilePath: 模型的文件地址
    :param paramFilePath:   模型参数的文件地址
    :return:
    """
    loadedModel = loadModelFromPkl(pickleFilePath)
    if paramFilePath is not None:
        loadedModel.load_state_dict(torch.load(paramFilePath))
    return loadedModel


def saveModelParamToLocal(model, output_file):
    """
    保存参数文件到本地
    :param model:
    :param output_file:
    :return:
    """
    if not output_file.endswith(".pth"):
        output_file = "".join([output_file, ".pth"])
    print(f"参数文件保存在{output_file}")
    torch.save(model.state_dict(), output_file)


def loadModelFromPkl(pickleFilePath):
    with open(pickleFilePath, 'rb') as file:
        loadedModel = pickle.load(file)
    return loadedModel


from Objects.OOrtProtocol import OOrtProtocol


def recvDataFromServer(connectSocket: socket):
    """
    从服务器接受bytes数据
    :param connectSocket:
    :return:
    """
    dataBytes = b''
    dataRecv = connectSocket.recv(1024)
    while dataRecv:
        dataBytes = dataBytes + dataRecv
        if dataBytes.endswith(OOrtProtocol.end_symbol_b):
            dataBytes = dataBytes.replace(OOrtProtocol.end_symbol_b, b"")
            break
        dataRecv = connectSocket.recv(1024)
    return dataBytes


def recvFileFromServer(connectSocket: socket, savePath, saveName, file_size):
    """
    从服务器接受文件
    :param connectSocket:
    :param savePath:
    :return:
    """
    size = len(os.listdir(savePath))
    saveName = saveName.replace("*", str(size))
    output_file_path = os.path.join(savePath, saveName)
    received_size = 0
    with open(output_file_path, 'wb') as file:
        while received_size < file_size:
            # 接收数据块
            data = connectSocket.recv(min(1024, file_size - received_size))
            if not data:
                break  # 如果没有数据了，跳出循环
            received_size += len(data)
            file.write(data)

    print(f"File received: {output_file_path}")


def sendFileToServer(fileName: str, clientSocket: socket):
    """
    从服务器下发文件到客户端
    :param fileName: 文件的完整路径名
    :param clientSocket: 客户端
    :return:
    """
    with open(fileName, "rb") as file:
        clientSocket.sendfile(file)


def sendParamsToServer(modelName, jobName, paramPath, clientSocket):
    """
    把模型发给服务器
    :param jobName: 工作的名字
    :param paramPath: 参数地址
    :param modelName:
    :param clientSocket:
    :return:
    """
    if not paramPath.endswith(".pth"):
        paramPath = "".join([paramPath, ".pth"])
    # 发配置文件
    sendFileToServer(paramPath, clientSocket)


def sendStrContentToTarget(dataContent: str, clientSocket: socket):
    """
    发送控制命令到目标
    :param dataContent: 数据内容
    :param clientSocket: 服务器与客户端的socket
    :return: Succeed(True) or Failed(False)
    """
    dataContent = "".join([dataContent, OOrtProtocol.end_symbol_s])
    clientSocket.sendall(dataContent.encode("utf-8"))


if __name__ == '__main__':
    elapsedTime, loss_2_sum = 0.0, 0.0
    # 训练过程中的临时目录
    temp_model_save_dir = "temp"
    if not os.path.exists(temp_model_save_dir):
        os.makedirs(temp_model_save_dir)
    print("临时目录已创建")
    # 新用户注册
    xmlConfigFilePath = "client_register.xml"
    xmlSize = get_file_size(xmlConfigFilePath)
    registerStr = OOrtProtocol.newInstance(
        command=OOrtProtocol.register_client,
        jobName="None",
        modelMame="None",
        pickleSize=0,
        param_size=0,
        xmlSize=xmlSize
    ).toString()
    configDict = parseXmlFileToDict(xmlConfigFilePath)
    connectSocket = connectToSever(configDict['target']['ip'], int(configDict['target']['port']))
    print("客户端开始监听服务器发来的信息...")
    # 发送注册命令到服务器
    sendStrContentToTarget(registerStr, connectSocket)
    while True:
        # 收配置
        dataBytes = recvDataFromServer(connectSocket)
        print(dataBytes.decode("utf-8"))
        oortPacket = OOrtProtocol.newInstanceByContext(dataBytes.decode("utf-8"))
        command = oortPacket.command
        jobName = oortPacket.jobName
        modelName = oortPacket.modelMame
        modelSize = oortPacket.pickleSize
        paramSize = oortPacket.param_size
        param_path = os.path.join(temp_model_save_dir, "param_trained")
        if command == OOrtProtocol.shutdown:
            print("Good Bye!")
            break
        elif command == OOrtProtocol.ask_for_client_config:
            print("========================服务器请求客户上传配置文件====================")
            sendFileToServer(xmlConfigFilePath, connectSocket)
            print("用户配置信息：")
            pprint(configDict)
        elif command == OOrtProtocol.param_from_server_to_client:
            print("========================从服务器加载模型和参数====================")
            # 收模型
            temp_model_pkl = os.path.join(temp_model_save_dir, f"{modelName}.pkl")
            recvFileFromServer(connectSocket, temp_model_save_dir, f"{modelName}.pkl", modelSize)
            # 收参数
            temp_param_pth = None
            if paramSize != 0:
                temp_param_pth = os.path.join(temp_model_save_dir, f"{modelName}.pth")
                recvFileFromServer(connectSocket, temp_model_save_dir, f"{modelName}.pth", paramSize)
            else:
                print("无现成checkpoint，需要初始训练...")
            # 加载模型
            model = loadModel(temp_model_pkl, temp_param_pth)
            print("模型加载成功...")
            """
                训练配置
            """
            # 设置超参数
            batch_size = 64
            learning_rate = 0.001
            epochs = 5
            # 设置优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # 设置损失函数
            criterion = torch.nn.CrossEntropyLoss()
            # 加载训练数据和测试数据
            train_loader, _ = getDataset(oortPacket.jobName, batch_size, 4, 3000)
            print("训练配置：")
            print(f"jobName: {jobName}")
            print(f"modelName: {modelName}")
            print(f"dataset: CIFAR-10")
            print(f"epochs: {epochs}")
            print(f"learning_rate: {learning_rate}")
            print(f"batch_size: {batch_size}")
            print(f"criterion: {criterion}")
            print(f"optimizer: {optimizer}")
            # 获得训练时间，和loss平方的和
            elapsedTime, loss_2_sum = train(model, train_loader, optimizer, criterion, epochs)
            print(f"训练耗时: {elapsedTime:.4f} seconds")
            print(f"loss_2_sum: {loss_2_sum:.4f}")
            # 保存模型参数到本地
            saveModelParamToLocal(model, output_file=param_path)
            # signature = calculate_file_signature("".join([param_path, ".pth"]))
            # print(f"参数文件的哈希值：{signature}")
        elif command == OOrtProtocol.aggregation_from_server_to_client:
            print("========================服务器请求聚合参数====================")
            param_size = get_file_size("".join([param_path, ".pth"]))
            aggregationNotice = OOrtProtocol.newInstance(
                command=OOrtProtocol.aggregation_from_client_to_server,
                jobName=jobName,
                modelMame=modelName,
                pickleSize=0,
                param_size=param_size,
                xmlSize=0,
                train_time=elapsedTime,
                loss_2_sum=loss_2_sum
            )
            sendStrContentToTarget(aggregationNotice.toString(), connectSocket)
            print(aggregationNotice.toString())
        elif command == OOrtProtocol.ask_for_params:
            print("========================向服务器传输参数====================")
            print(f"参数地址：{param_path}")
            sendParamsToServer(modelName, jobName,
                               paramPath=param_path, clientSocket=connectSocket, )
            print("========================传输完成====================")
        # elif command == OOrtProtocol.ask_for_metrics:
        #     print("========================向服务器传输训练时间和指标====================")
        #     metricsNotice = OOrtProtocol.newInstance(
        #         command=OOrtProtocol.aggregation_from_client_to_server,
        #         jobName=jobName,
        #         modelMame=modelName,
        #         pickleSize=0,
        #         param_size=param_size,
        #         xmlSize=0,
        #         train_time=elapsedTime,
        #         loss_2_sum=loss_2_sum
        #     )
        #     sendStrContentToTarget(metricsNotice.toString(), connectSocket)
        #     print(metricsNotice.toString())
