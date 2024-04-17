import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        Loss_gradient = F.l1_loss(gradient_A, gradient_B)
        return Loss_gradient

if __name__ == '__main__':
    x = torch.rand(12, 64, 128, 128).cuda()
    y = torch.rand(12, 64, 128, 128).cuda()

    loss = L_Grad()
    loss_num = loss(x, y)
    print(loss_num)