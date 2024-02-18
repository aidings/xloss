import torch
from xloss import PatchDiscriminator

if __name__ == '__main__':
    x = torch.randn(2,3, 512, 512)
    x = x.abs()
    x = x / x.max()
    print(x.max(), x.min())
    p = PatchDiscriminator()

    y = p(x)
    print(y.shape)

    g_loss = torch.mean(y)
    print(-g_loss)