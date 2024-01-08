import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics.pairwise as smp


def fed_avg(local_models):
    model_agg = copy.deepcopy(local_models[0].state_dict())
    for k in model_agg.keys():
        model_agg[k] = 0
        model_agg[k] = torch.div(model_agg[k], len(local_models))
    return model_agg


class NoiseDataset(Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise


def get_update(rank):
    """
    将相似度排名转换成对效用估计分布的更新
    :param rank: 排名
    :return: 效用估计分布的更新
    """
    print(rank)
    if len(rank) == 1:
        return [1]
    length = len(rank) - 1
    updates = [value / length for value in rank]
    print(updates)
    return updates


def get_rank(score, reverse=False):
    """
    对分数进行排序
    :param score:
    :param reverse:
    :return:
    """
    sorted_lst = sorted(score, reverse=reverse)
    ranks_dict = {value: index for index, value in enumerate(sorted_lst)}
    ranks = [ranks_dict[value] for value in score]
    return ranks


class SelectMain(object):

    def __init__(self, num_channels=0, width=0, height=0, device="cpu"):
        """
        :param num_channels: 图片通道数
        :param device:
        """
        self.num_samples = 100
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.similarity = None
        self.client_info = OrderedDict()
        self.expect_duration = 100000000
        self.device = device

    def setConfig(self, num_channels, image_dims):
        self.num_channels = num_channels
        self.width = image_dims[0]
        self.height = image_dims[1]

    def register_client(self, clientId):
        """
        传入clientId注册
        :param clientId:
        :return:
        """
        if clientId not in self.client_info:
            self.client_info[clientId] = {}
            self.client_info[clientId]['alpha'] = 1
            self.client_info[clientId]['beta'] = 1
            self.client_info[clientId]['duration'] = 0

    def update_client(self, clientId, duration, update):
        """
        更新客户信息
        :param clientId: 客户的id，int
        :param duration: 客户局部训练时间
        :param update: 参数更新值
        :return:
        """
        self.client_info[clientId]['duration'] = duration
        self.client_info[clientId]['alpha'] += update
        self.client_info[clientId]['beta'] += (1 - update)

    def calculate_similarity(self, global_model: torch.nn.Module, local_models_param: OrderedDict):
        """
        计算相似性全局模型和本地模型的相似性
        :param global_model: 全局模型
        :param local_models_param:
        :return: 返回相似性分数
        """
        local_models = []
        for param in local_models_param:
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(param)
            local_models.append(local_model)
        dataset = NoiseDataset([self.num_channels, self.width, self.height], self.num_samples)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        global_op = []
        global_model.eval()
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                global_op.append(global_model(data).cpu().numpy().flatten())
        local_ops = []
        for local_model in local_models:
            local_model.eval()
            local_op = []
            for data in loader:
                data = data.to(self.device)
                with torch.no_grad():
                    local_op.append(local_model(data).cpu().numpy().flatten())
            local_ops.append(local_op)
        similarity = []
        for local_op in local_ops:
            tmp = np.vstack((np.array(local_op).flatten(), np.array(global_op).flatten()))
            sims = smp.cosine_distances(tmp)
            similarity.append(sims[0][1])
        self.similarity = similarity
        return similarity

    def get_score(self, client_id):
        """
        获得对应客户端的id求得他的分数，用于排名
        :param client_id:
        :return: 评分
        """
        alpha = self.client_info[client_id]['alpha']
        beta = self.client_info[client_id]['beta']
        score = np.random.beta(alpha, beta)
        return score

    def select_participant(self, select_num, durations):
        """
        客户端选择算法
        :param select_num: 多少个客户端
        :param durations: 客户端对应的训练时间
        :return:
        """
        client_list = list(self.client_info.keys())
        if len(durations) != 0:
            expect_duration = np.median(durations) * 1.1

            if self.expect_duration != 100000000:
                self.expect_duration = int(expect_duration + self.expect_duration) / 2
            else:
                self.expect_duration = expect_duration
        scores = []

        for clients in client_list:
            score = self.get_score(clients)
            if self.client_info[clients]['duration'] > self.expect_duration:
                score *= math.sqrt(self.expect_duration / self.client_info[clients]['duration'])
            scores.append(score)
        rank = get_rank(scores, True)
        indices = [index for index, num in enumerate(rank) if num < select_num]
        select_client = [client_list[i] for i in indices]
        print(select_client)
        return select_client


if __name__ == '__main__':
    print(get_rank([3, 1, 2, 4, 9, 0, 100]))
