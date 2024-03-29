import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class AddFeature(nn.Module):
    def __init__(self):
        super(AddFeature, self).__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.step2 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.step3 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.step4 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x, step):
        if step == 1:
            x = self.step1(x)
            x = self.bn1(x)
        elif step == 2:
            x = self.step2(x)
            x = self.bn2(x)
        elif step == 3:
            x = self.step3(x)
            x = self.bn3(x)
        elif step == 4:
            x = self.step4(x)
            x = self.bn4(x)

        return x


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

        self.add = AddFeature()

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

        feature_map = self.regressor(O4)

        return self.Th(self.pred(O4)), feature_map


if __name__ == '__main__':
    x = torch.rand(4, 60, 128, 128)
    y = torch.rand(4, 1, 128, 128)
    net = UNet()
    print(hasattr(net, 'regressor'))
    net.train()
    x, _ = net(y, x)
    print(x.shape)
    print(_.shape)

#  不需要加，直接比loss

