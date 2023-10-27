import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

train_data = FashionMNIST(root='../data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0)

# 获得一个batch的数据
for step, (X, y) in enumerate(train_loader):
    if step > 0:
        break

batch_x = X.squeeze().numpy()
batch_y = y.numpy()
class_label = train_data.classes
print(f'label: {class_label}')
print(f'The size of batch in train data: {X.shape}')  # torch.Size([64, 1, 224, 224])
print(f'The size of batch batch_x: {batch_x.shape}')  # (64, 224, 224)

# 可视化一个batch的数据
plt.figure(figsize=(12, 5))
for i in np.arange(len(y)):
    plt.subplot(4, 16, i + 1)
    plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
