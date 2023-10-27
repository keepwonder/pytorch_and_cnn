import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import GoogLeNet, Inception

from tqdm import tqdm


def test_data_process():
    # 数据集路径
    root_train = './data/test'
    # 定义数据集处理方法
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    # 加载数据集
    test_data = ImageFolder(root_train, transform=transform)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print(f'测试的准确率为：{test_acc}')


if __name__ == '__main__':
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)
