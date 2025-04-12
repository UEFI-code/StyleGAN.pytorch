import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, kernelSize = 3, stride = 3, dim = 1024, padding = 3):
        super(Discriminator, self).__init__()
        self.activation = nn.ReLU()
        self.down1 = nn.Conv2d(3, dim, kernelSize, stride, padding, bias=False)
        self.down2 = nn.Conv2d(dim, dim, kernelSize, stride, padding, bias=False)
        self.down3 = nn.Conv2d(dim, dim, kernelSize, stride, padding, bias=False)
        self.down4 = nn.Conv2d(dim, dim, kernelSize, stride, padding, bias=False)
        self.down5 = nn.Conv2d(dim, dim, kernelSize, stride, padding, bias=False)
        self.down6 = nn.Conv2d(dim, dim, kernelSize, stride, padding, bias=False)
        self.windup = nn.Conv2d(dim, 1, kernelSize, stride, padding, bias=False)

    def forward(self, x):
        x = self.down1(x)
        x = self.activation(x)
        x = self.down2(x)
        x = self.activation(x)
        x = self.down3(x)
        x = self.activation(x)
        x = self.down4(x)
        x = self.activation(x)
        # x = self.down5(x)
        # x = self.activation(x)
        # print(x.shape)
        #x = self.down6(x)
        x = self.windup(x) # the output is 4x4
        return x

if __name__ == '__main__':
    discriminator = Discriminator()
    image = torch.randn(1, 3, 192, 192)
    print(discriminator(image).shape)