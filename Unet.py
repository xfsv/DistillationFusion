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
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvolutionLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = int(in_channels / 2)
        denseblock = []

        denseblock += [ConvolutionLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvolutionLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class UNet(nn.Module):

    def __init__(self, original_model):
        super(UNet, self).__init__()

        image_channel = 2
        out_channel = 16
        nb_filter = [24, 48, 64, 80]
        block = DenseBlock
        kernel_size= 3

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Conv = ConvLayer(
            in_channels=image_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=2
        )

        # encoder
        self.layer_1 = original_model.layer_1
        self.layer_2 = original_model.layer_2
        self.layer_3 = original_model.layer_3
        self.layer_4 = original_model.layer_4
        self.layer_5 = original_model.layer_5

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)

        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        # generate
        self.generate = InvertedResidual(
            in_channels=nb_filter[0] + out_channel,
            out_channels=1,
            stride=1,
            expand_ratio=1
        )

        # loss function
        self.mse = nn.MSELoss()
        self.grad_loss = L_Grad()
        self.ssim_loss = ssim
        self.intensity_loss = L_Intensity()
        self.total_loss = TotalLoss()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        x = self.Conv(x)
        # encoder
        x1_0 = self.layer_1(x)
        x2_0 = self.layer_2(x1_0)
        x3_0 = self.layer_3(x2_0)
        x4_0 = self.layer_4(x3_0)
        x5_0 = self.layer_5(x4_0)

        # decoder
        x1_1 = self.DB1_1(torch.cat([x2_0, self.up(x3_0)], 1))

        x2_1 = self.DB2_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x1_2 = self.DB1_2(torch.cat([x2_0, x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x2_2 = self.DB2_2(torch.cat([x3_0, x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([x2_0, x1_1, x1_2, self.up(x2_2)], 1))

        y = self.up(torch.cat([self.up(x1_3), x1_0], dim=1))
        outputs = self.generate(y)

        return outputs


if __name__ == '__main__':
    x = torch.rand(4, 1, 288, 288).to(device)
    y = torch.rand(4, 1, 288, 288).to(device)

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
    # print(original_model)
    net = UNet(original_model).to(device)
    net.train()
    x = net(x, y)
    print(x.shape)

#  不需要加，直接比loss
