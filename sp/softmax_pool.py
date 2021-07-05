import torch
import torch.nn as nn
import torch.nn.functional as F


# Dynamic parameter
# https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/4
# class MyModule(nn.Module):
#     def __init__(self):
#         # you need to register the parameter names earlier
#         self.register_parameter('weight', None)
#
#     def reset_parameters(self, input):
#         self.weight = nn.Parameter(input.new(input.size()).normal_(0, 1))
#
#     def forward(self, input):
#         if self.weight is None:
#             self.reset_parameters(input)
#         return self.weight @ input

class GlobalSoftMaxPool2d(nn.Module):
    def __init__(self, num_classes: int = None):
        super(GlobalSoftMaxPool2d, self).__init__()
        self.register_parameter('gsp', None)
        self.softmax = nn.Softmax(dim=-1)
        self.num_classes = num_classes
        if self.num_classes is None:
            self.conv = None
            self.bn = None

    def reset_parameters(self, x):
        param_size = [x.size(1) if self.num_classes is None else self.num_classes, x.size(2) * x.size(3)]
        self.gsp = nn.Parameter(
            torch.randn(*param_size, device=x.device, dtype=x.dtype)
        )
        if self.num_classes is not None:
            self.conv = nn.Conv2d(x.size(1), self.num_classes, kernel_size=1).to(x.device)
            self.bn = nn.BatchNorm2d(self.num_classes).to(x.device)

    def forward(self, x):
        """
        Expected input: (batch, n_c, h, w)
        """
        if self.gsp is None:
            self.reset_parameters(x)
        if self.num_classes is not None:
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)  # Make generic
        batch, n_c, h, w = x.size()
        x = x.view(batch, n_c, h * w)
        x = self.softmax(self.gsp) * x
        x = x.view(batch, n_c, h, w)
        return x.sum([2, 3])  # out shape -> (batch, n_c)
