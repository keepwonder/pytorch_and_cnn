import torch
from torch import nn
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{torch.cuda.get_device_name()}')


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.ReLU = nn.ReLU()

        # 路线1: 1x1卷积
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路线2: 1x1卷积, 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 路线3: 1x1卷积, 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 路线4： 3x3最大池化, 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x


if __name__ == '__main__':
    model = GoogLeNet(Inception).to(device)
    print(summary(model, (1, 224, 224)))

"""
device:NVIDIA GeForce GTX 1660 Ti
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           3,200
              ReLU-2         [-1, 64, 112, 112]               0
         MaxPool2d-3           [-1, 64, 56, 56]               0
            Conv2d-4           [-1, 64, 56, 56]           4,160
              ReLU-5           [-1, 64, 56, 56]               0
            Conv2d-6          [-1, 192, 56, 56]         110,784
              ReLU-7          [-1, 192, 56, 56]               0
         MaxPool2d-8          [-1, 192, 28, 28]               0
            Conv2d-9           [-1, 64, 28, 28]          12,352
             ReLU-10           [-1, 64, 28, 28]               0
           Conv2d-11           [-1, 96, 28, 28]          18,528
             ReLU-12           [-1, 96, 28, 28]               0
           Conv2d-13          [-1, 128, 28, 28]         110,720
             ReLU-14          [-1, 128, 28, 28]               0
           Conv2d-15           [-1, 16, 28, 28]           3,088
             ReLU-16           [-1, 16, 28, 28]               0
           Conv2d-17           [-1, 32, 28, 28]          12,832
             ReLU-18           [-1, 32, 28, 28]               0
        MaxPool2d-19          [-1, 192, 28, 28]               0
           Conv2d-20           [-1, 32, 28, 28]           6,176
             ReLU-21           [-1, 32, 28, 28]               0
        Inception-22          [-1, 256, 28, 28]               0
           Conv2d-23          [-1, 128, 28, 28]          32,896
             ReLU-24          [-1, 128, 28, 28]               0
           Conv2d-25          [-1, 128, 28, 28]          32,896
             ReLU-26          [-1, 128, 28, 28]               0
           Conv2d-27          [-1, 192, 28, 28]         221,376
             ReLU-28          [-1, 192, 28, 28]               0
           Conv2d-29           [-1, 32, 28, 28]           8,224
             ReLU-30           [-1, 32, 28, 28]               0
           Conv2d-31           [-1, 96, 28, 28]          76,896
             ReLU-32           [-1, 96, 28, 28]               0
        MaxPool2d-33          [-1, 256, 28, 28]               0
           Conv2d-34           [-1, 64, 28, 28]          16,448
             ReLU-35           [-1, 64, 28, 28]               0
        Inception-36          [-1, 480, 28, 28]               0
        MaxPool2d-37          [-1, 480, 14, 14]               0
           Conv2d-38          [-1, 192, 14, 14]          92,352
             ReLU-39          [-1, 192, 14, 14]               0
           Conv2d-40           [-1, 96, 14, 14]          46,176
             ReLU-41           [-1, 96, 14, 14]               0
           Conv2d-42          [-1, 208, 14, 14]         179,920
             ReLU-43          [-1, 208, 14, 14]               0
           Conv2d-44           [-1, 16, 14, 14]           7,696
             ReLU-45           [-1, 16, 14, 14]               0
           Conv2d-46           [-1, 48, 14, 14]          19,248
             ReLU-47           [-1, 48, 14, 14]               0
        MaxPool2d-48          [-1, 480, 14, 14]               0
           Conv2d-49           [-1, 64, 14, 14]          30,784
             ReLU-50           [-1, 64, 14, 14]               0
        Inception-51          [-1, 512, 14, 14]               0
           Conv2d-52          [-1, 160, 14, 14]          82,080
             ReLU-53          [-1, 160, 14, 14]               0
           Conv2d-54          [-1, 112, 14, 14]          57,456
             ReLU-55          [-1, 112, 14, 14]               0
           Conv2d-56          [-1, 224, 14, 14]         226,016
             ReLU-57          [-1, 224, 14, 14]               0
           Conv2d-58           [-1, 24, 14, 14]          12,312
             ReLU-59           [-1, 24, 14, 14]               0
           Conv2d-60           [-1, 64, 14, 14]          38,464
             ReLU-61           [-1, 64, 14, 14]               0
        MaxPool2d-62          [-1, 512, 14, 14]               0
           Conv2d-63           [-1, 64, 14, 14]          32,832
             ReLU-64           [-1, 64, 14, 14]               0
        Inception-65          [-1, 512, 14, 14]               0
           Conv2d-66          [-1, 128, 14, 14]          65,664
             ReLU-67          [-1, 128, 14, 14]               0
           Conv2d-68          [-1, 128, 14, 14]          65,664
             ReLU-69          [-1, 128, 14, 14]               0
           Conv2d-70          [-1, 256, 14, 14]         295,168
             ReLU-71          [-1, 256, 14, 14]               0
           Conv2d-72           [-1, 24, 14, 14]          12,312
             ReLU-73           [-1, 24, 14, 14]               0
           Conv2d-74           [-1, 64, 14, 14]          38,464
             ReLU-75           [-1, 64, 14, 14]               0
        MaxPool2d-76          [-1, 512, 14, 14]               0
           Conv2d-77           [-1, 64, 14, 14]          32,832
             ReLU-78           [-1, 64, 14, 14]               0
        Inception-79          [-1, 512, 14, 14]               0
           Conv2d-80          [-1, 112, 14, 14]          57,456
             ReLU-81          [-1, 112, 14, 14]               0
           Conv2d-82          [-1, 128, 14, 14]          65,664
             ReLU-83          [-1, 128, 14, 14]               0
           Conv2d-84          [-1, 288, 14, 14]         332,064
             ReLU-85          [-1, 288, 14, 14]               0
           Conv2d-86           [-1, 32, 14, 14]          16,416
             ReLU-87           [-1, 32, 14, 14]               0
           Conv2d-88           [-1, 64, 14, 14]          51,264
             ReLU-89           [-1, 64, 14, 14]               0
        MaxPool2d-90          [-1, 512, 14, 14]               0
           Conv2d-91           [-1, 64, 14, 14]          32,832
             ReLU-92           [-1, 64, 14, 14]               0
        Inception-93          [-1, 528, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         135,424
             ReLU-95          [-1, 256, 14, 14]               0
           Conv2d-96          [-1, 160, 14, 14]          84,640
             ReLU-97          [-1, 160, 14, 14]               0
           Conv2d-98          [-1, 320, 14, 14]         461,120
             ReLU-99          [-1, 320, 14, 14]               0
          Conv2d-100           [-1, 32, 14, 14]          16,928
            ReLU-101           [-1, 32, 14, 14]               0
          Conv2d-102          [-1, 128, 14, 14]         102,528
            ReLU-103          [-1, 128, 14, 14]               0
       MaxPool2d-104          [-1, 528, 14, 14]               0
          Conv2d-105          [-1, 128, 14, 14]          67,712
            ReLU-106          [-1, 128, 14, 14]               0
       Inception-107          [-1, 832, 14, 14]               0
       MaxPool2d-108            [-1, 832, 7, 7]               0
          Conv2d-109            [-1, 256, 7, 7]         213,248
            ReLU-110            [-1, 256, 7, 7]               0
          Conv2d-111            [-1, 160, 7, 7]         133,280
            ReLU-112            [-1, 160, 7, 7]               0
          Conv2d-113            [-1, 320, 7, 7]         461,120
            ReLU-114            [-1, 320, 7, 7]               0
          Conv2d-115             [-1, 32, 7, 7]          26,656
            ReLU-116             [-1, 32, 7, 7]               0
          Conv2d-117            [-1, 128, 7, 7]         102,528
            ReLU-118            [-1, 128, 7, 7]               0
       MaxPool2d-119            [-1, 832, 7, 7]               0
          Conv2d-120            [-1, 128, 7, 7]         106,624
            ReLU-121            [-1, 128, 7, 7]               0
       Inception-122            [-1, 832, 7, 7]               0
          Conv2d-123            [-1, 384, 7, 7]         319,872
            ReLU-124            [-1, 384, 7, 7]               0
          Conv2d-125            [-1, 192, 7, 7]         159,936
            ReLU-126            [-1, 192, 7, 7]               0
          Conv2d-127            [-1, 384, 7, 7]         663,936
            ReLU-128            [-1, 384, 7, 7]               0
          Conv2d-129             [-1, 48, 7, 7]          39,984
            ReLU-130             [-1, 48, 7, 7]               0
          Conv2d-131            [-1, 128, 7, 7]         153,728
            ReLU-132            [-1, 128, 7, 7]               0
       MaxPool2d-133            [-1, 832, 7, 7]               0
          Conv2d-134            [-1, 128, 7, 7]         106,624
            ReLU-135            [-1, 128, 7, 7]               0
       Inception-136           [-1, 1024, 7, 7]               0
AdaptiveAvgPool2d-137           [-1, 1024, 1, 1]               0
         Flatten-138                 [-1, 1024]               0
          Linear-139                   [-1, 10]          10,250
================================================================
Total params: 5,927,850
Trainable params: 5,927,850
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 69.44
Params size (MB): 22.61
Estimated Total Size (MB): 92.24
----------------------------------------------------------------
"""