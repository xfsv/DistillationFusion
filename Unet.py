import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class ChangeChannel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChangeChannel, self).__init__()
        self.DW = ConvBNReLU(in_channel, in_channel, kernel_size=3, stride=1, groups=in_channel)
        self.PW = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.DW(x)
        x = self.PW(x)

        return x


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),

            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),

            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()

        self.Down = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.Up = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.C1 = Conv(1, 64)
        self.D1 = DownSample(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSample(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSample(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSample(512)
        self.C5 = Conv(512, 1024)

        self.U1 = UpSample(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSample(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSample(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSample(128)
        self.C9 = Conv(128, 64)

        self.regressor = nn.Sequential(
            nn.Conv2d(64, 60, kernel_size=3, stride=1, padding=1)
        )

        self.change = ChangeChannel
        self.mse = nn.MSELoss()

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, feature=None):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        loss_list = []
        if feature is not None and self.training is not None:
            feature.detach()
            r1 = self.C1(feature)
            r2 = self.C2(self.D1(r1))
            r3 = self.C3(self.D2(r2))
            r4 = self.C4(self.D3(r3))
            y1 = self.C5(self.D4(r4))

            o1 = self.C6(self.U1(y1, r4))
            loss_list.append(self.mse(o1, O1))
            o2 = self.C7(self.U2(o1, r3))
            loss_list.append(self.mse(o2, O2))
            o3 = self.C8(self.U3(o2, r2))
            loss_list.append(self.mse(o3, O3))
            o4 = self.C9(self.U4(o3, r1))
            loss_list.append(self.mse(o4, O4))

        feature_map = self.regressor(O4)
        if feature is not None and self.training is not None:
            return self.Th(self.pred(O4)), feature_map, loss_list
        else:
            return self.Th(self.pred(O4))


if __name__ == '__main__':
    x = torch.rand(4, 1, 128, 128)
    y = torch.rand(4, 1, 128, 128)
    net = UNet()
    net.train()
    x, _ = net(y, x)

#  不需要加，直接比loss
