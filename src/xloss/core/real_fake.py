import torch
import torch.nn.functional as F
from ..utils import RealFakeCriterion

class RealFakeLoss:
    def __init__(self, loss_type: RealFakeCriterion = RealFakeCriterion.HINGE):
        self.loss_func = self.__hinge_d_loss if loss_type == RealFakeCriterion.HINGE else self.__vanilla_d_loss

    @staticmethod 
    def __hinge_d_loss(logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    @staticmethod 
    def __vanilla_d_loss(logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
        return d_loss
    
    def __call__(self, logits_real, logits_fake):
        return self.loss_func(logits_real, logits_fake)