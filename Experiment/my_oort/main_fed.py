import copy

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from models.Fed import FedAvg
from models.Mbn import MobileNet
from models.Nets import CNNMnist
from models.Res import ResNet
from models.Update import LocalUpdate
from models.test import test_img
from my_oort.models.Alex import AlexNet
from myoort import Oort, get_client_feedback
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid


def write_to_file(num, fname):
    with open("{}.txt".format(fname), "a") as f:
        # 将epoch和loss写入文件
        f.write(f"{num}\n")


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:0')

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

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    oo = Oort(args=args)
    acc = []
    with open("oort.txt", 'w') as file:
        file.truncate()
    with open("oortloss.txt", 'w') as file:
        file.truncate()
    for i in range(args.num_users):
        oo.register_client(i, len(dict_users[i]))
    for epoch in range(args.epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        clients_list = oo.select_participant(m, epoch + 1)
        for idx in clients_list:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            feedback = get_client_feedback(loss, 10, len(dict_users[idx]), epoch + 1)
            oo.update_client_util(idx, feedback)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)
        net_glob.eval()
        acc_train, loss_train1 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test1 = test_img(net_glob, dataset_test, args)
        acc.append(acc_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        write_to_file(acc_test, 'oort')
        write_to_file(loss_avg, 'oortloss')

    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc)), acc)
    plt.ylabel('train_acc')
    plt.savefig('oort.png')
