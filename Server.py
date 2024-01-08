import os.path
import socket
from collections import OrderedDict
from pprint import pprint
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models.BlackNet_cifar10
from Objects import ClientObject
from Objects.ClientObject import clientConstruction
from Utils.ShaTool import calculate_file_signature
from Utils.XMLTool import parseXmlFileToDict, parseXmlStringToDict
from Utils.FileUtil import get_file_size
from Utils.ModelTool import saveModelParamToLocal, loadModelFromPkl
from clientselection.my_oort.models.Fed import FedAvg
from Objects.OOrtProtocol import OOrtProtocol


def sendFileToClient(fileName: str, clientSocket: socket):
    """
    从服务器下发文件到客户端
    :param fileName: 文件的完整路径名
    :param clientSocket: 客户端
    :return:
    """
    with open(fileName, "rb") as file:
        clientSocket.sendfile(file)


def sendStrContentToClient(dataContent: str, clientSocket: socket):
    """
    发送控制命令到客户端
    :param dataContent: 数据内容
    :param clientSocket: 服务器与客户端的socket
    :return: Succeed(True) or Failed(False)
    """
    dataContent = "".join([dataContent, OOrtProtocol.end_symbol_s])
    clientSocket.sendall(dataContent.encode("utf-8"))


def recvDataFromSocket(src: socket, speed: int = 1024):
    """
    该函数从一个socket中获得数据
    :param src: 从一个源socket获得数据
    :param speed: 每一次获得多少个字节，默认1024
    :return:
    """
    contentStr = ""
    line = src.recv(speed)
    while line:
        contentStr = "".join((contentStr, line.decode("utf-8")))
        if contentStr.endswith(OOrtProtocol.end_symbol_s):
            contentStr = contentStr.replace(OOrtProtocol.end_symbol_s, "")
            break
        line = src.recv(speed)
    return contentStr


rounds = 5


def endSignal():
    """
    学习是否结束判断
    :param flag:
    :return:
    """
    global rounds
    rounds -= 1
    if rounds == -1:
        return False
    return True


def connectToClient(ipAddress, port):
    # 创建客户端socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.bind((request.srcIp, request.srcPort))
    # 连接到服务器
    client_socket.connect((ipAddress, port))
    return client_socket


def serialize_and_save_model(model, output_file):
    import pickle
    # 将模型序列化为字节流
    serialized_model = pickle.dumps(model)
    # 保存序列化后的模型到文件
    with open(output_file, 'wb') as file:
        file.write(serialized_model)
    print(f"Serialized model saved to {output_file}")


def sendModelToClient(modelName, jobName, paramPath, clientSocket, beforeNoticeFunc):
    """
    把模型发给客户端
    :param jobName: 工作的名字
    :param paramPath: 参数地址
    :param beforeNoticeFunc:
    :param modelName:
    :param clientSocket:
    :return:
    """
    serializedDir = "models/serialized/"
    serializedPath = None
    if modelName == "BlackNet_cifar10":
        from models.BlackNet_cifar10 import BlackNet
        serializedPath = f"{serializedDir}BlackNet_cifar10.pkl"
        serialize_and_save_model(models.BlackNet_cifar10.get_model_instance(), serializedPath)
    elif modelName == "BlackNet_mnist":
        from models.BlackNet_mnist import BlackNet
        serializedPath = f"{serializedDir}BlackNet_mnist.pkl"
        serialize_and_save_model(models.BlackNet_mnist.get_model_instance(), serializedPath)
    elif modelName == "BlackNet_fashion_mnist":
        from models.BlackNet_fashion_mnist import BlackNet
        serializedPath = f"{serializedDir}BlackNet_fashion_mnist.pkl"
        serialize_and_save_model(models.BlackNet_fashion_mnist.get_model_instance(), serializedPath)
    # TODO: 发送一些配置信息
    modelSize = get_file_size(serializedPath)
    paramSize = get_file_size(paramPath)
    content = OOrtProtocol.newInstance(
        command=OOrtProtocol.param_from_server_to_client,
        jobName=jobName,
        modelMame=modelName,
        pickleSize=modelSize,
        param_size=paramSize,
        xmlSize=0
    ).toString()
    # 发配置文件
    beforeNoticeFunc(content, clientSocket)
    sendFileToClient(serializedPath, clientSocket)
    if paramSize != 0:
        sendFileToClient(paramPath, clientSocket)


def dispatchModelToSingleClient(clientManager, client, jobName, modelName, paramPath):
    """
    发送模型命令和模型参数到客户端
    :param jobName: 工作的名字
    :param clientManager:
    :param client: 客户端
    :param modelName: 模型的名字
    :param paramPath: 参数路径
    :return:
    """
    clientSocket = clientManager.getClientSocketByIpCard(client.ipCard())
    # TODO: 如果已经有先前已经聚合好的参数就发，如果没有客户端需要根据本地数据训练
    # 发模型和参数
    sendModelToClient(modelName, jobName, paramPath, clientSocket, sendStrContentToClient)


def dispatchModelToClients(clientManager, clients, jobName, modelName, paramPath):
    """
    从服务器分发模型到客户端
    :param clientManager: 客户端管理器
    :param clients: 选到的客户端
    :param modelName: 任务使用到的模型
    :param paramPath: 上一次训练得到的模型地址，初始为None
    :return:
    """
    for client in clients:
        dispatchModelToSingleClient(clientManager, client, jobName, modelName, paramPath)


def recvSpecificBytesFileFromClient(connectSocket: socket, savePath, saveName, file_size):
    """
    从客户端接受文件
    :param file_size:
    :param saveName:
    :param connectSocket:
    :param savePath:
    :return:
    """
    if not os.path.exists(savePath):
        os.makedirs(savePath)
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


def getParamFromClient(clientManager, client, jobName, modelName, saveDir="temp/aggregation"):
    """
    从客户端获得参数
    :param modelName: 模型的名字
    :param saveDir: 聚合的临时目录
    :param jobName: 工作的名字
    :param clientManager: 客户端管理者
    :param client: 客户端
    :return 参数文件存储在服务器本地的名字
    """
    connectSocket = clientManager.getClientSocketByIpCard(client.ipCard())
    aggregationNotice = OOrtProtocol.newInstance(
        command=OOrtProtocol.aggregation_from_server_to_client,
        jobName=jobName,
        modelMame=modelName,
        pickleSize=0,
        param_size=0,
        xmlSize=0
    ).toString()
    sendStrContentToClient(aggregationNotice, connectSocket)
    clientNoticeList = recvDataFromSocket(connectSocket).split(" ")
    param_size = int(clientNoticeList[4])
    param_name = f"{jobName}_{client.getName()}.pth"
    train_time = float(clientNoticeList[6])
    loss_2_sum = float(clientNoticeList[7])
    askForParams = OOrtProtocol.newInstance(
        command=OOrtProtocol.ask_for_params,
        jobName=jobName,
        modelMame=modelName,
        pickleSize=0,
        param_size=0,
        xmlSize=0
    ).toString()
    sendStrContentToClient(askForParams, connectSocket)
    recvSpecificBytesFileFromClient(connectSocket, saveDir, saveName=f"{param_name}",
                                    file_size=param_size)
    return [param_name, train_time, loss_2_sum]


def calculate_metrics(labels, predictions):
    """
    计算召回率、精度、F1 Score 和混淆矩阵
    """
    tp = torch.sum((labels == 1) & (predictions == 1)).item()
    tn = torch.sum((labels == 0) & (predictions == 0)).item()
    fp = torch.sum((labels == 0) & (predictions == 1)).item()
    fn = torch.sum((labels == 1) & (predictions == 0)).item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    confusion_mat = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())

    return recall, precision, f1, confusion_mat


def evalModel(model: torch.nn.Module, param: OrderedDict, test_loader: DataLoader):
    """
    验证模型的效果
    :param test_loader: 测试数据
    :param model: 模型
    :param param: 参数
    :return:
    """
    model.load_state_dict(param)
    result = {'loss': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'confusion_matrix': None}

    # 设置模型为评估模式
    model.eval()

    with torch.no_grad():  # 在评估阶段不计算梯度
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        for i, data in enumerate(test_loader):
            X, Y = data
            # 前向传播
            y_hat = model(X)
            # 计算损失
            loss = F.cross_entropy(y_hat, Y)
            result['loss'] += loss.item()
            # 计算精度
            _, predicted = torch.max(y_hat, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

            # 收集标签和预测结果以计算召回率和精度
            all_labels.extend(Y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        result['accuracy'] = correct / total

        # 转换为 PyTorch 的 Tensor
        all_labels = torch.tensor(all_labels)
        all_predictions = torch.tensor(all_predictions)

        # 计算召回率、精度、F1 Score 和混淆矩阵
        result['recall'], result['precision'], result['f1'], result['confusion_matrix'] = calculate_metrics(all_labels,
                                                                                                            all_predictions)

    print(f'Test Loss: {result["loss"] / len(test_loader):.4f}')
    print(f'Test Accuracy: {result["accuracy"] * 100:.2f}%')
    print(f'Test Recall: {result["recall"]:.4f}')
    print(f'Test Precision: {result["precision"]:.4f}')
    print(f'Test F1 Score: {result["f1"]:.4f}')
    print(f'Confusion Matrix:\n{result["confusion_matrix"]}')

    return result


# 使用示例
# evalModel(my_model, my_parameters, my_test_loader)


from Utils.DatasetTool import getDataset


def delete_all_files_in_directory(directory):
    """
    删除目录中的所有文件
    :param directory: 目录路径
    """
    try:
        # 获取目录中的所有文件
        file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # 删除每个文件
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)

        print(f'所有文件已成功删除: {directory}')
    except Exception as e:
        print(f'删除文件时发生错误: {e}')


def collectMetric(clientManager, clients, jobName, modelName, param):
    pass

clients_index = 0

class ParameterServer:
    TRANSFER_END = "****"

    def __init__(self, host, port):
        """
        服务器初始化
        :param port: 服务器监听的端口
        """
        # 用户管理员
        self.clientManager = ClientManager()
        # 服务器运行端口
        self.ENABLE_RUNNING = True
        # 服务器的ip地址
        self.host = host
        # 服务器监听的端口
        self.port = port
        # 创建一个socket对象
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 将socket绑定到指定的主机和端口
        self.serverSocket.bind((self.host, self.port))
        # 开始监听传入的连接,100为最大的连接数
        self.serverSocket.listen(100)

    def start(self):
        print(f"Server listening on {self.host}:{self.port}")
        while self.ENABLE_RUNNING:
            # 等待客户端连接
            client_socket, client_address = self.serverSocket.accept()
            print(f"Accepted connection from {client_address}")
            # 解析客户端发送的数据
            srcData = recvDataFromSocket(client_socket)
            request = OOrtProtocol.newInstanceByContext(srcData)
            # 解析客户端请求的服务，分配对应的函数
            self.requestRedirect(request, client_socket)

    def registerClient(self, client: ClientObject, clientSocket: socket):
        """
        新来到的客户端加入用户列表
        :return:
        """
        self.clientManager.registerFreshClient(client, clientSocket)

    def requestRedirect(self, request: OOrtProtocol, clientSocket: socket):
        """
        根据客户请求数据进行重定向
        :param request:
        :param clientSocket: 客户端的socket
        :param clientRequestData: 客户发来的请求数据
        :return: 逻辑处理函数
        """
        if request.command == OOrtProtocol.register_client:
            self.processRegister(clientSocket, request)

        elif request.command == OOrtProtocol.job_submit:
            self.processJobSubmit(clientSocket, request)

    def processJobSubmit(self, clientSocket: socket, request: OOrtProtocol):
        """
        处理工作上交逻辑
        :param clientSocket:
        :param request:
        :return:
        """
        jobSubmitResponse = OOrtProtocol.newInstance(
            command=OOrtProtocol.ask_for_job_config,
            jobName=OOrtProtocol.default,
            modelMame=OOrtProtocol.default,
            pickleSize=0,
            param_size=0,
            xmlSize=0
        )
        # 发送一个response
        sendStrContentToClient(jobSubmitResponse.toString(), clientSocket)
        # 等待接受用户发送的文件
        recvSpecificBytesFileFromClient(clientSocket, "jobs_config", "job_demo.xml", request.xmlSize)
        print("工作保存在jobs_config\\job_demo.xml")
        clientConfigDict = parseXmlFileToDict(os.path.join("jobs_config", "job_demo.xml"))
        self.handleJob(clientConfigDict)

    def processRegister(self, clientSocket: socket, request: OOrtProtocol):
        global clients_index
        """
        处理客户注册
        :param clientSocket:
        :param request:
        :return:
        """
        registerResponse = OOrtProtocol.newInstance(
            command=OOrtProtocol.ask_for_client_config,
            jobName=OOrtProtocol.default,
            modelMame=OOrtProtocol.default,
            pickleSize=0,
            param_size=0,
            xmlSize=0
        )
        # 发送一个response
        sendStrContentToClient(registerResponse.toString(), clientSocket)
        # 等待接受用户发送的文件
        saveDir = "clients_config"
        num = len(os.listdir(saveDir))
        saveName = f"client_demo{num}.xml"
        recvSpecificBytesFileFromClient(clientSocket, "clients_config", saveName, request.xmlSize)
        print(f"收到用户的配置文件，文件保存在 clients_config\\{saveName}")
        clientConfigDict = parseXmlFileToDict(os.path.join("clients_config", saveName))
        clientConfigDict['src']['ip'], clientConfigDict['src']['port'] = clientSocket.getpeername()
        self.registerClient(clientConstruction(clients_index, clientConfigDict), clientSocket)
        clients_index += 1

    def handleJob(self, jobConfig):
        """
        该函数用来处理工作
        :param jobConfig: 工作配置
        :return:
        """
        # 输出任务的相关信息
        global rounds
        rounds = int(jobConfig["job"]["hyper_parameters"]["rounds"])
        jobName = jobConfig["job"]["job_name"]
        workerNum = int(jobConfig["job"]["workers"])
        dataset = jobConfig["job"]["dataset"]
        modelName = jobConfig["job"]["models"]["model"]

        print("\n==================================job submitted================================")
        print(f"jobName: {jobName}")
        print(f"workerNum: {workerNum}")
        print(f"dataset: {dataset}")
        print(f"model: {modelName}")
        print(f"rounds: {rounds}")
        weightPath = f"models/checkpoints/{modelName}.pth"
        # 切换任务输入
        self.clientManager.fixSelectMachine(jobName)
        durations_last = []
        loss_2_sum_last = []
        for round in range(rounds):
            print(f"\n==================================第{round}轮================================")
            print("\n====客户端选择====")
            # TODO: 客户端选择未开发完全
            # clients = self.clientManager.getCandidateClients(int(workerNum))
            clients = self.clientManager.getCandidateClients(workerNum, mode="YuanPi", durations=durations_last, loss_2_sum_list=loss_2_sum_last)
            for i, client in enumerate(clients):
                print(f"rank_{i} {client}")
            print("\n====分发模型====")
            if not os.path.exists(weightPath):
                print("checkpoint暂不存在")
                dispatchModelToClients(self.clientManager, clients, jobName, modelName, None)
            else:
                print("checkpoint已存在")
                dispatchModelToClients(self.clientManager, clients, jobName, modelName, weightPath)

            print("\n====等待客户端训练====")
            paramSavedDir = "temp\\aggregation"
            configVecs = []
            for client in clients:
                configVecs.append(getParamFromClient(self.clientManager, client, jobName, modelName, paramSavedDir))
            print("客户端参数下载完毕......")
            print("\n====parameter聚合=====")
            paramList = []
            time_list = []
            loss_2_sum_list = []
            for i in range(len(clients)):
                paramList.append(os.path.join(paramSavedDir, configVecs[i][0]))
                time_list.append(configVecs[i][1])
                loss_2_sum_list.append(configVecs[i][2])
            dict_list = []
            for i in paramList:
                dict_list.append(torch.load(i))
            aggregatedParam = FedAvg(dict_list)
            saveModelParamToLocal(aggregatedParam, weightPath)
            print(f"聚合后的参数存储地址:{weightPath}")
            print("\n====评估聚合效果====")
            _, test_loader = getDataset(dataset)
            evalModel(loadModelFromPkl(f"models/serialized/{modelName}.pkl"),
                      torch.load(weightPath), test_loader)
            # 计算模型间的相似度
            serverModel = loadModelFromPkl(f"models/serialized/{modelName}.pkl")
            serverModel.load_state_dict(torch.load(weightPath))
            similarity = self.clientManager.selectMachine.calculate_similarity(serverModel, dict_list)
            print(similarity)
            # 计算更新值
            updates = get_update(get_rank(similarity, True))
            # 更新排名
            for i in range(len(clients)):
                self.clientManager.updateRank(clients[i].getId(), time_list[i], loss_2_sum_list[i], updates[i])
            durations_last = time_list
            loss_2_sum_last = loss_2_sum_list
            delete_all_files_in_directory(paramSavedDir)


from clientselection.my_oort.myts import SelectMain, get_rank, get_update


class ClientManager:
    def __init__(self):
        """
        客户端管理者，管理新到的客户端以及从可选用户中进行客户端选择
        """
        self.onlineClients = []
        self.clientSocketMap = {}
        self.idToClient = {}
        self.selectMachine = SelectMain()

    def fixSelectMachine(self, jobName):
        num_channels, image_dim = 0, 0
        if jobName.lower() == "cifar_10" or jobName.lower() == "cifar_100":
            num_channels, image_dim = 3, [32, 32]
        elif jobName.lower() == "mnist" or jobName.lower() == "fashion_mnist":
            num_channels, image_dim = 1, [28, 28]
        self.selectMachine.setConfig(num_channels, image_dim)

    def registerFreshClient(self, freshClient: ClientObject, freshSocket:socket):
        """
        新用户到来需要注册信息；
        :param freshSocket:
        :param freshClient:
        :return:
        """
        print("==================================client registered================================")
        if freshClient not in self.onlineClients:
            self.onlineClients.append(freshClient)
            self.clientSocketMap[f"{freshClient.ipCard()}"] = freshSocket
            self.idToClient[freshClient.getId()] = freshClient
            print(f"{freshClient.ipCard()} 已注册...")
            print(f"在线用户: {len(self.clientSocketMap)}")
            self.selectMachine.register_client(freshClient.getId())
        else:
            print("用户已存在...")
            try:
                peer_address = self.clientSocketMap[freshClient.ipCard()].getpeername()
                print(f"Peer address: {peer_address}")
            except socket.error as e:
                print(f"Error getting peer address: {e}")

    def getOnlineClients(self):
        """
        获得当前在线的用户列表
        :return:
        """
        return self.onlineClients, self.clientSocketMap

    def getCandidateClients(self, workerNum, mode="random", durations=None, loss_2_sum_list=None):
        """
        执行客户端选择
        :param loss_2_sum_list: 客户端训练损失和
        :param durations: 客户端的训练延迟
        :param mode: 客户端选择算法，random/yuanpi/oort
        :param workerNum:需要多少个用户参与本轮训练
        :return: 返回被选用的客户端列表
        """
        if workerNum > len(self.onlineClients):
            print("目前服务器暂未注册足够数量的客户...")
            return self.onlineClients
        if mode.lower() == "random":
            import random
            return random.sample(self.onlineClients, workerNum)
        elif mode.lower() == "yuanpi":
            return [self.idToClient[id] for id in self.selectMachine.select_participant(workerNum, durations)]
        elif mode.lower() == "oort":
            return None
        return None

    def getClientSocketByIpCard(self, ipCard):
        """
        客户端的身份证
        :param ipCard: 身份证
        :return: 对应的客户端对象
        """
        return self.clientSocketMap.get(ipCard)

    def updateRank(self, clientId, time, loss_2_sum, update):
        """
        更新clientId的排名
        :param update:
        :param clientId:
        :param time:
        :param loss_2_sum:
        :return:
        """
        self.selectMachine.update_client(clientId, time, update)


if __name__ == '__main__':
    config = parseXmlFileToDict("server_config.xml")
    print("服务器初始化成功...")
    ps = ParameterServer(config['server']['ip'], int(config['server']['port']))
    ps.start()

