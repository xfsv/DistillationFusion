import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_ssim import ssim
from Grad_loss import L_Grad
from Intensity_loss import L_Intensity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

        self.ssim_loss = ssim
        self.grad_loss = L_Grad()
        self.intensity_loss = L_Intensity()

    def forward(self, image_a, image_b):
        loss1 = 1 - self.ssim_loss(image_a, image_b)
        loss2 = self.grad_loss(image_a, image_b)
        loss3 = self.intensity_loss(image_a, image_b)

        return loss1 + loss2 + loss3


if __name__ == '__main__':
    x = torch.rand(1, 1, 480, 640).to(device)
    y = torch.rand(1, 1, 480, 640).to(device)

    loss = TotalLoss()
    loss_num = loss(x, y)
    print(loss_num)