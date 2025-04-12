import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, kernelSize = 3, stride = 3, dim = 256, padding = 3):
        super(Discriminator, self).__init__()
        self.activation = nn.LeakyReLU()
        self.down1 = nn.Conv2d(3, dim, kernelSize, stride, padding)
        self.down2 = nn.Conv2d(dim, dim, kernelSize, stride, padding)
        self.down3 = nn.Conv2d(dim, dim, kernelSize, stride, padding)
        self.down4 = nn.Conv2d(dim, dim, kernelSize, stride, padding)
        self.down5 = nn.Conv2d(dim, dim, kernelSize, stride, padding)
        self.down6 = nn.Conv2d(dim, dim, kernelSize, stride, padding)
        self.windup = nn.Conv2d(dim, 1, kernelSize, stride, padding)

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
        x = torch.mean(x, dim=(2, 3))
        return x

if __name__ == '__main__':
    discriminator = Discriminator()
    image = torch.randn(1, 3, 192, 192)
    print(discriminator(image).shape)