import math
import random
from collections import OrderedDict
import numpy as np


class Oort(object):
    def __init__(self, args):
        self.exploitClients = None
        self.training_round = 0
        self.client_info = OrderedDict()
        self.unexplored = set()
        self.args = args
        self.prefer_duration = 1000
        self.exploration = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.sample_window = self.args.sample_window
        self.exploreClients = []

    def register_client(self, clientId, size):
        if clientId not in self.client_info:
            self.client_info[clientId] = {}
            self.client_info[clientId]['reward'] = float(size)
            self.client_info[clientId]['duration'] = 0
            self.client_info[clientId]['last_sample'] = self.training_round
            self.client_info[clientId]['count'] = 0
            self.client_info[clientId]['status'] = True
            self.unexplored.add(clientId)

    def update_client_util(self, clientId, feedbacks):
        self.client_info[clientId]['reward'] = feedbacks['reward']
        self.client_info[clientId]['duration'] = feedbacks['duration']
        self.client_info[clientId]['last_sample'] = feedbacks['last_sample']
        self.client_info[clientId]['count'] += 1
        self.client_info[clientId]['status'] = feedbacks['status']
        self.unexplored.discard(clientId)

    def select_participant(self, sample_size, cur_time):
        util = {}
        self.training_round = cur_time
        client_list = list(self.client_info.keys())
        reward = []
        select_client = []

        # Exploitation #1: Calculate client utility.
        for clients in client_list:
            if self.client_info[clients]['reward'] > 0:
                tmp = self.client_info[clients]['reward']
                reward.append(tmp)

        max_reward, min_reward, range_reward, avg_reward, clip_value = get_norm(reward)

        for client in client_list:
            if self.client_info[client]['count'] > 0:
                client_reward = min(self.client_info[client]['reward'], clip_value)
                client_utility = (client_reward - min_reward) / float(range_reward)
                # Temporal uncertainty.
                client_utility += math.sqrt(0.1 * math.log(cur_time) / self.client_info[client]['last_sample'])

                client_duration = self.client_info[client]['duration']
                if client_duration > self.prefer_duration:
                    client_utility *= ((float(self.prefer_duration) / max(1e-4,
                                                                          client_duration)) ** self.args.round_penalty)
                util[client] = client_utility

        # Exploitation #2
        client_lakes = list(util.keys())
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        explore_len = min(int(sample_size * (1.0 - self.exploration)), len(client_lakes))

        sorted_util = sorted(util, key=util.get, reverse=True)

        if len(sorted_util) != 0:
            cut_off_util = util[sorted_util[explore_len]] * self.args.cut_off_util
            for client in sorted_util:
                if util[client] < cut_off_util:
                    break
                select_client.append(client)

            sum_util = max(1e-4, float(sum([util[key] for key in select_client])))
            print(explore_len)
            select_client = list(
                np.random.choice(select_client, explore_len, p=[util[key] / sum_util for key in select_client],
                                 replace=False))

        self.exploitClients = select_client

        if len(self.unexplored) > 0:
            init_reward = {}
            for client in self.unexplored:
                init_reward[client] = self.client_info[client]['reward']
                client_duration = self.client_info[client]['duration']

                if client_duration > self.prefer_duration:
                    init_reward[client] *= (
                            (float(self.prefer_duration) / max(1e-4, client_duration)) ** self.args.round_penalty)

            wait_to_explore = min(len(self.unexplored), sample_size - len(select_client))
            unexplore_set = sorted(init_reward, key=init_reward.get, reverse=True)[
                            :min(int(self.sample_window * wait_to_explore), len(init_reward))]

            unexplored_util = float(sum([init_reward[key] for key in unexplore_set]))

            select_unexplore = list(np.random.choice(unexplore_set, wait_to_explore,
                                                     p=[init_reward[key] / max(1e-4, unexplored_util) for key in
                                                        unexplore_set], replace=False))
            self.exploreClients = select_unexplore
            select_client = select_client + select_unexplore
        else:
            self.exploration_min = 0.
            self.exploration = 0.

        while len(select_client) < sample_size:
            random_client = random.choice(client_list)
            if random_client not in select_client:
                select_client.append(random_client)

        return select_client


def get_client_feedback(loss, duration, sample_size, sampled_round):
    reward = math.sqrt(loss * sample_size)
    feedbacks = {
        'reward': reward,
        'duration': duration,
        'status': True,
        'last_sample': sampled_round
    }
    return feedbacks


def get_norm(aList, clip_bound=0.98, thres=1e-4):
    if len(aList) == 0:
        return 0, 0, 0, 0, 0
    aList.sort()
    clip_value = aList[min(int(len(aList) * clip_bound), len(aList) - 1)]

    _max = max(aList)
    _min = min(aList) * 0.999
    _range = max(_max - _min, thres)
    _avg = sum(aList) / max(1e-4, float(len(aList)))

    return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
