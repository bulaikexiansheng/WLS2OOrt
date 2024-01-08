from pprint import pformat


class ClientObject:
    def __init__(self, id, ipAddress, port):
        self.id = id
        self.port = port
        self.ipAddress = ipAddress
        self.name = "black"

    def ipCard(self):
        return ":".join([self.ipAddress, str(self.port)])

    def getId(self):
        return self.id

    def getName(self):
        return self.name

    def getPort(self):
        return self.port

    def getIpAddress(self):
        return self.ipAddress

    def __hash__(self):
        # 自定义哈希值的计算方式
        return hash(self.ipAddress)

    def __eq__(self, other):
        if isinstance(other, ClientObject):
            return self.ipCard() == other.ipCard()
        return False

    def __str__(self):
        return pformat(self.__dict__)


def clientConstruction(client_id, clientDataDict: dict):
    return ClientObject(client_id, clientDataDict["src"]["ip"], clientDataDict["src"]["port"])


if __name__ == '__main__':
    print(ClientObject("1", 2))
