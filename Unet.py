import torch
from typing import Optional, Tuple, Union, Dict
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from Total_loss import TotalLoss
from pytorch_ssim import ssim
from Grad_loss import L_Grad
from Intensity_loss import L_Intensity
from MobileModule import MobileViT, model_config, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(ConvBNReLU, self).__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownConvBNRuLU(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3,
                 dilation=1, flag: bool = True):
        super(DownConvBNRuLU, self).__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x):
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),

            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
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
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.Up = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.Up(up)
        return torch.cat((x, r), dim=1)


class UNet(nn.Module):

    def __init__(self, original_model):
        super(UNet, self).__init__()

        image_channel = 2
        out_channel = 16
        self.Conv = ConvLayer(
            in_channels=image_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=2
        )

        self.layer_1 = original_model.layer_1
        self.layer_2 = original_model.layer_2
        self.layer_3 = original_model.layer_3
        self.layer_4 = original_model.layer_4
        self.layer_5 = original_model.layer_5

        self.U1 = UpSample(80, 64)
        self.C6 = Conv(128, 64)
        self.U2 = UpSample(64, 48)
        self.C7 = Conv(96, 48)
        self.U3 = UpSample(48, 24)
        self.C8 = Conv(48, 24)
        self.U4 = UpSample(24, 16)
        self.C9 = Conv(32, 16)

        self.down1 = DownConvBNRuLU(64, 128)
        self.down2 = DownConvBNRuLU(128, 256)
        self.down3 = DownConvBNRuLU(256, 512)
        self.conv = ConvBNReLU(1, 64)
        self.mse = nn.MSELoss()
        self.grad_loss = L_Grad()
        self.ssim_loss = ssim
        self.intensity_loss = L_Intensity()
        self.total_loss = TotalLoss()

        self.Th = torch.nn.LeakyReLU()
        self.pred = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        x = self.Conv(x)  # N, 16, 64, 64
        R1 = self.layer_1(x)  # N, 16, 64, 64
        R2 = self.layer_2(R1)  # N, 24, 32, 32
        R3 = self.layer_3(R2)  # N, 48, 16, 16
        R4 = self.layer_4(R3)  # N, 64, 8, 8
        R5 = self.layer_5(R4)  # N, 80, 4, 4

        O1 = self.C6(self.U1(R5, R4))  # N, 64, 8, 8
        O2 = self.C7(self.U2(O1, R3))  # N, 48, 16, 16
        O3 = self.C8(self.U3(O2, R2))  # N, 24, 32, 32
        O4 = self.C9(self.U4(O3, R1))  # N, 16, 64, 64

        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return self.Th(self.pred(x))


if __name__ == '__main__':
    x = torch.rand(4, 1, 128, 128).to(device)
    y = torch.rand(4, 1, 128, 128).to(device)

    config = model_config.get_config("xx_small")
    original_model = MobileViT.MobileViT(config, num_classes=1000)
    weight_path = r'\MobileViT\mobilevit_xxs.pt'
    weights_dict = torch.load(weight_path, map_location=device)
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "classifier" in k:
            del weights_dict[k]
    original_model.load_state_dict(weights_dict, strict=False)
    for name, para in original_model.named_parameters():
        if "layer_" in name:
            para.requires_grad_(False)
            print("yes")
        else:
            print("no")
    # print(original_model)
    net = UNet(original_model).to(device)
    net.train()
    x = net(x, y)
    print(x.shape)

#  不需要加，直接比loss
