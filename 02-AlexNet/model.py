import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6 * 6 * 256, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(x, 0.5)

        x = self.ReLU(self.f2(x))
        x = F.dropout(x, 0.5)

        x = self.f3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    print(summary(model, (1, 224, 224)))

# 卷积层参数量计算
# Channel_out * (kernel_w *kernel_h * Channel_in) + bias(=Channel_out)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 96, 55, 55]          11,712   
              ReLU-2           [-1, 96, 55, 55]               0
         MaxPool2d-3           [-1, 96, 27, 27]               0
            Conv2d-4          [-1, 256, 27, 27]         614,656
              ReLU-5          [-1, 256, 27, 27]               0
         MaxPool2d-6          [-1, 256, 13, 13]               0
            Conv2d-7          [-1, 384, 13, 13]         885,120
              ReLU-8          [-1, 384, 13, 13]               0
            Conv2d-9          [-1, 384, 13, 13]       1,327,488
             ReLU-10          [-1, 384, 13, 13]               0
           Conv2d-11          [-1, 256, 13, 13]         884,992
             ReLU-12          [-1, 256, 13, 13]               0
        MaxPool2d-13            [-1, 256, 6, 6]               0
          Flatten-14                 [-1, 9216]               0
           Linear-15                 [-1, 4096]      37,752,832
             ReLU-16                 [-1, 4096]               0
           Linear-17                 [-1, 4096]      16,781,312
             ReLU-18                 [-1, 4096]               0
           Linear-19                   [-1, 10]          40,970
             ReLU-20                   [-1, 10]               0
================================================================
Total params: 58,299,082
Trainable params: 58,299,082
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.20
Forward/backward pass size (MB): 11.05
Params size (MB): 222.39
Estimated Total Size (MB): 233.64
----------------------------------------------------------------
"""