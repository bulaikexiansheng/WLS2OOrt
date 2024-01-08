import collections
import pickle
import torch


def loadModelFromPkl(pickleFilePath):
    with open(pickleFilePath, 'rb') as file:
        loadedModel = pickle.load(file)
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
    if type(model) == torch.nn.Module:
        torch.save(model.state_dict(), output_file)
    elif type(model) == collections.OrderedDict:
        torch.save(model, output_file)
    return output_file
