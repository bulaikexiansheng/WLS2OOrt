import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    """
    calculate model acc and loss
    :param net_g: golbal model
    :param datatest:
    :param args:
    :return:
    """
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        net_g = net_g.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(datatest)
    accuracy = 100.00 * correct / len(datatest)
    return accuracy, test_loss
