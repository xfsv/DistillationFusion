import torch
import torch.nn as nn
from torch.nn import functional as F


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B):
        Loss_intensity = F.l1_loss(image_A, image_B)
        return Loss_intensity


if __name__ == '__main__':
    x = torch.rand(1, 1, 480, 640)
    y = torch.rand(1, 1, 480, 640)
    loss = L_Intensity()
    loss_num = loss(x, y)
    print(loss_num)

