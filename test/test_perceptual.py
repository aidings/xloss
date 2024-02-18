import torch
from xloss import PerceptualLoss

if __name__ == "__main__":
    loss_func = PerceptualLoss(reduction='none')

    real = torch.randn(2, 1, 30, 30)
    fake = torch.randn(2, 1, 30, 30)

    loss = loss_func(real, fake)

    print(loss, loss.size())
    

