from torch import nn


class BlackNet(nn.Module):
    def __init__(self):
        super(BlackNet, self).__init__()
        self.input_dims = 784  # 28x28 for MNIST
        self.hidden_dims = 128
        self.output_dims = 10  # 10 classes for MNIST
        # MLP with one hidden layer
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dims, self.hidden_dims), nn.ReLU(),
            nn.Linear(self.hidden_dims, 64), nn.ReLU(),
            nn.Linear(64, self.output_dims)
        )

        # 初始化每一层的参数
        def init_weights(module):
            if type(module) == nn.Module:
                nn.init.zeros_(module.bias)
                nn.init.normal_(module.weight, 0.0, 1)

        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


def get_model_instance():
    return BlackNet()
