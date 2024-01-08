import socket


def connectToSever(targetIp, targetPort):
    # 创建客户端socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.bind((request.srcIp, request.srcPort))
    # 连接到服务器
    client_socket.connect((targetIp, targetPort))
    return client_socket
