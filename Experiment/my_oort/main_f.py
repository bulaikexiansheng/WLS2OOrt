import copy

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from models.Alex import AlexNet
from models.Fed import FedAvg
from models.Mbn import MobileNet
from models.Nets import CNNMnist
from models.Res import ResNet
from models.Update import LocalUpdate
from models.test import test_img
from myts import mytsa, get_update, get_rank
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid


def write_to_file(num, fname):
    # 打开文件，以追加模式写入数据
    with open("{}.txt".format(fname), "a") as f:
        # 将epoch和loss写入文件
        f.write(f"{num}\n")


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'mbn' and args.dataset == 'cifar':
        net_glob = MobileNet(args=args).to(args.device)
    elif args.model == 'cnn':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'alex' and args.dataset == 'cifar':
        net_glob = AlexNet().to(args.device)
    elif args.model == 'res' and args.dataset == 'cifar100':
        net_glob = ResNet(100).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    with open("ours.txt", 'w') as file:
        file.truncate()
    with open("oursloss.txt", 'w') as file:
        file.truncate()
    # training
    loss_train = []
    acc = []
    mytt = mytsa(args)
    for i in range(args.num_users):
        mytt.register_client(i, len(dict_users[i]))
    for epoch in range(args.epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        clients_list = mytt.select_participant(m, [])
        for idx in clients_list:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        similarity = mytt.calculate_similarity(copy.deepcopy(net_glob), w_locals)
        updates = get_update(get_rank(similarity, True))
        for idx, user in enumerate(clients_list):
            mytt.update_client(user, 1, updates[idx])
        # copy weight to net_glob

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train1 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        acc.append(acc_test)
        write_to_file(acc_test, 'ours')
        write_to_file(loss_avg, 'oursloss')
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc)), acc)
    plt.ylabel('train_acc')
    plt.savefig('ours.png')
