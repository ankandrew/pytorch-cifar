import torch.nn as nn


class Patcher(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module = None):
        super(Patcher, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        x = self.m1(x)
        if self.m2 is not None:
            x = self.m2(x)
        return x
