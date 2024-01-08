import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms


def load_cifar(batch_size=64, samplers=None):
    """
    本地训练数据和测试数据
    :param samplers: 如果是None,那就整个数据集都返回；如果不是，就随机采样N张
    :param shuffle: 是否随机采样数据集
    :param batch_size:
    :return:
    """
    random_sampler = None
    if samplers is not None:
        random_sampler = SubsetRandomSampler(range(samplers))
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train, batch_size, sampler=random_sampler)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=False)
    return train_loader, test_loader


def load_fashion_mnist(batch_size=64, samplers=None):
    """
    返回 Fashion MNIST 数据集的训练和测试 DataLoader
    :param batch_size: 每个批次的样本数
    :param shuffle: 是否对数据进行洗牌
    :return: train_loader, test_loader
    """
    random_sampler = None
    if samplers is not None:
        random_sampler = SubsetRandomSampler(range(samplers))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 加载训练数据集
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)

    # 加载测试数据集
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_mnist(batch_size=64, samplers=None):
    """
    返回 MNIST 数据集的训练和测试 DataLoader
    :param samplers:
    :param batch_size: 每个批次的样本数
    :param shuffle: 是否对数据进行洗牌
    :return: train_loader, test_loader
    """
    random_sampler = None
    if samplers is not None:
        random_sampler = SubsetRandomSampler(range(samplers))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 加载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)

    # 加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_cifar100(batch_size=64, samplers=None):
    """
    返回 CIFAR-100 数据集的训练和测试 DataLoader
    :param samplers:
    :param batch_size: 每个批次的样本数
    :param shuffle: 是否对数据进行洗牌
    :return: train_loader, test_loader
    """
    random_sampler = None
    if samplers is not None:
        random_sampler = SubsetRandomSampler(range(samplers))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 加载训练数据集
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler)

    # 加载测试数据集
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def getDataset(dataset: str, batch_size=64, num_workers=4, samplers=None):
    """
    返回训练数据和测试数据
    :param dataset:  数据集的名字
    :param batch_size: 批处理大小
    :param num_workers: 暂时别管
    :return:
    """
    if dataset.lower() == "cifar_10":
        return load_cifar(batch_size,  samplers=samplers)
    elif dataset.lower() == "mnist":
        return load_mnist(batch_size,  samplers=samplers)
    elif dataset.lower() == "fashion_mnist":
        return load_fashion_mnist(batch_size,  samplers=samplers)
    elif dataset.lower() == "cifar_100":
        return load_cifar100(batch_size,  samplers=samplers)
