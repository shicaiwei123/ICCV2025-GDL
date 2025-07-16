'''保留有意义的测试代码'''
import torchvision.transforms as tt
import torch
import numpy as np
rotaion = tt.Compose([tt.RandomRotation(30)])


def transform_test():
    from PIL import Image

    img_pil = Image.open(
        "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg").convert(
        'RGB')
    img_r = rotaion(img_pil)
    img_b = rotaion(img_pil)
    img_r.show()
    img_b.show()


def dropout_test():
    def dropout(X, drop_prob):
        X = X.float()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        # 这种情况下把全部元素都丢弃
        if keep_prob == 0:
            return torch.zeros_like(X)
        mask = (torch.rand(X.shape) < keep_prob).float()

        return mask * X / keep_prob

    W1 = torch.tensor(np.random.normal(0, 0.01, size=(16, 30)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(30, requires_grad=True)

    X = torch.arange(16).view(1, 16)
    X = X.float()
    X = X.view(-1, 16)
    a = torch.matmul(X, W1) + b1
    b = dropout(a, 0.5)
    print(1)


def pool_test():
    from models.surf_baseline import SURF_Baseline
    from configuration.config_baseline_multi import args
    import torch.nn.functional as tnf

    a = torch.tensor([np.float32(x) for x in range(9)])
    a = a.reshape(3, -1)
    a = torch.unsqueeze(a, 0)
    b = tnf.adaptive_avg_pool2d(a, (2, 2))
    print(b)