from torch import nn


class BlackNet(nn.Module):
    def __init__(self):
        super(BlackNet, self).__init__()
        self.input_dims = 32 * 32 * 3  # CIFAR-10 图片的维度
        self.hidden_dims = 220
        self.output_dims = 10  # CIFAR-10 类别数
        # 一层隐藏层的多层感知机
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dims, self.hidden_dims), nn.ReLU(),
            nn.Linear(self.hidden_dims, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.output_dims)
        )

        # 每一层参数的初始化
        def init_weights(module):
            if type(module) == nn.Module:
                nn.init.zeros_(module.bias)
                nn.init.normal_(module.weight, 0.0, 1)

        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


def get_model_instance():
    return BlackNet()
