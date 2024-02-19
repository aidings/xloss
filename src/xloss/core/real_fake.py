import torch
import torch.nn.functional as F
from ..utils import RealFakeCriterion, LossReduction, Reduction

class RealFakeLoss:
    def __init__(self, reduction:LossReduction='mean', loss_type: RealFakeCriterion = RealFakeCriterion.HINGE):
        self.loss_func = self.__hinge_d_loss if loss_type == RealFakeCriterion.HINGE else self.__vanilla_d_loss
        self.reduction = Reduction(reduction=reduction) 

    def __hinge_d_loss(self, logits_real, logits_fake):
        loss_real = self.reduction(F.relu(1. - logits_real))
        loss_fake = self.reduction(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def __vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            self.reduction(torch.nn.functional.softplus(-logits_real)) +
            self.reduction(torch.nn.functional.softplus(logits_fake)))
        return d_loss
    
    def __call__(self, logits_real, logits_fake):
        return self.loss_func(logits_real, logits_fake)