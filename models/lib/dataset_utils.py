import torchvision.datasets as td
import torchvision.transforms as tt
from torch.utils.data.sampler import WeightedRandomSampler
import torch


def analyze_data_code():
    '''
    分析pytorch data 相关代码
    :return:
    '''
    mnist = td.MNIST(root="/home/shicaiwei/data/domain_adda", transform=tt.ToTensor())

    # 是迭代器,但是不能以next访问
    # a=mnist.next()

    # 这个对象的默认方法怎么访问?不像是transform 和model 不接受输入.

    # 不是生成器,因为可以无限次访问.
    # for data, label in mnist:
    #     print(label)

    # SAMPLE
    weights = []
    num_samples = 0
    for data, label in mnist:
        num_samples += 1
        if label == 0:
            weights.append(20)
        elif label == 1 or label == 2 or label == 4:
            weights.append(10)
        else:
            weights.append(1.6)
    sampler = WeightedRandomSampler(weights, num_samples=128, replacement=True)

    loader = torch.utils.data.DataLoader(mnist, batch_size=32, sampler=sampler)

    for batch_idx, (data, target) in enumerate(loader):
        print(target.tolist())




if __name__ == '__main__':
    analyze_data_code()
