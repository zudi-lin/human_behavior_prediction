import torch
import torch.nn as nn
import torch.nn.functional as F
from .non_local import NONLocalBlock2D

class GameModelPooling(nn.Module):
    def __init__(self, in_planes=2, out_planes=1, kernels=8, mode='max_pool', 
                 bias=True, non_local=False, residual=False):
        super().__init__()

        assert mode in ['max_pool', 'avg_pool']
        self.mode = mode
        self.residual = residual

        self.conv1 = nn.Conv2d(in_planes, kernels, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(kernels*3, kernels, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(kernels*3, kernels, kernel_size=1, bias=bias)
        self.conv4 = nn.Conv2d(kernels*3, out_planes, kernel_size=1, bias=bias)

        # normalization layers
        self.bn1 = nn.BatchNorm2d(kernels)
        self.bn2 = nn.BatchNorm2d(kernels)
        self.bn3 = nn.BatchNorm2d(kernels)

        self.non_local = non_local
        if self.non_local: 
            self.non_local_layer = NONLocalBlock2D(kernels)
        # weights initialization
        self._init_weight()

    def _forward_residual(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = x
        x = self.pool_and_cat(x)
        x = F.relu(self.bn2(self.conv2(x)) + y)
        y = x
        x = self.pool_and_cat(x)
        x = F.relu(self.bn3(self.conv3(x)) + y)
        if self.non_local: 
            x = self.non_local_layer(x)
        x = self.pool_and_cat(x)
        x = self.conv4(x)
        return x

    def _forward_plain(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_and_cat(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_and_cat(x)
        x = F.relu(self.bn3(self.conv3(x)))
        if self.non_local: 
            x = self.non_local_layer(x)
        x = self.pool_and_cat(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        if self.residual:
            x = self._forward_residual(x)
        else:
            x = self._forward_plain(x)

        B, C, H, W = x.size()
        # run softmax and get marginal distribution
        x = x.view(B,C,-1)
        x = F.softmax(x, dim=2)
        x = x.view(B, C, H, W)
        return x

    def pool_and_cat(self, x):
        if self.mode == 'max_pool':
            x_row = F.max_pool2d(x, kernel_size=(1,3))
            x_col = F.max_pool2d(x, kernel_size=(3,1))
        else:
            x_row = F.avg_pool2d(x, kernel_size=(1,3))
            x_col = F.avg_pool2d(x, kernel_size=(3,1))
        x_row = x_row.expand(x.size())
        x_col = x_row.expand(x.size())
        y = torch.cat((x, x_row, x_col), dim=1)
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

