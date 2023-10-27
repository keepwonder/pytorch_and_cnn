import torch
from torch import nn
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{torch.cuda.get_device_name()}')


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, stride=1):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self, Residual):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, stride=1),
            Residual(64, 64, use_1conv=False, stride=1)
        )

        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128, use_1conv=False, stride=1)
        )

        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2),
            Residual(256, 256, use_1conv=False, stride=1)
        )

        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2),
            Residual(512, 512, use_1conv=False, stride=1)
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == '__main__':
    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))


"""
device:NVIDIA GeForce GTX 1660 Ti
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           3,200
              ReLU-2         [-1, 64, 112, 112]               0
       BatchNorm2d-3         [-1, 64, 112, 112]             128
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,928
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,928
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
         Residual-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,928
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,928
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
         Residual-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,856
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,584
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,320
             ReLU-25          [-1, 128, 28, 28]               0
         Residual-26          [-1, 128, 28, 28]               0
           Conv2d-27          [-1, 128, 28, 28]         147,584
      BatchNorm2d-28          [-1, 128, 28, 28]             256
             ReLU-29          [-1, 128, 28, 28]               0
           Conv2d-30          [-1, 128, 28, 28]         147,584
      BatchNorm2d-31          [-1, 128, 28, 28]             256
             ReLU-32          [-1, 128, 28, 28]               0
         Residual-33          [-1, 128, 28, 28]               0
           Conv2d-34          [-1, 256, 14, 14]         295,168
      BatchNorm2d-35          [-1, 256, 14, 14]             512
             ReLU-36          [-1, 256, 14, 14]               0
           Conv2d-37          [-1, 256, 14, 14]         590,080
      BatchNorm2d-38          [-1, 256, 14, 14]             512
           Conv2d-39          [-1, 256, 14, 14]          33,024
             ReLU-40          [-1, 256, 14, 14]               0
         Residual-41          [-1, 256, 14, 14]               0
           Conv2d-42          [-1, 256, 14, 14]         590,080
      BatchNorm2d-43          [-1, 256, 14, 14]             512
             ReLU-44          [-1, 256, 14, 14]               0
           Conv2d-45          [-1, 256, 14, 14]         590,080
      BatchNorm2d-46          [-1, 256, 14, 14]             512
             ReLU-47          [-1, 256, 14, 14]               0
         Residual-48          [-1, 256, 14, 14]               0
           Conv2d-49            [-1, 512, 7, 7]       1,180,160
      BatchNorm2d-50            [-1, 512, 7, 7]           1,024
             ReLU-51            [-1, 512, 7, 7]               0
           Conv2d-52            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-53            [-1, 512, 7, 7]           1,024
           Conv2d-54            [-1, 512, 7, 7]         131,584
             ReLU-55            [-1, 512, 7, 7]               0
         Residual-56            [-1, 512, 7, 7]               0
           Conv2d-57            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-58            [-1, 512, 7, 7]           1,024
             ReLU-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
         Residual-63            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-64            [-1, 512, 1, 1]               0
          Flatten-65                  [-1, 512]               0
           Linear-66                   [-1, 10]           5,130
================================================================
Total params: 11,178,378
Trainable params: 11,178,378
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 61.45
Params size (MB): 42.64
Estimated Total Size (MB): 104.28
----------------------------------------------------------------
"""