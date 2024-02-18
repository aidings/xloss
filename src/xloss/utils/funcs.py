import torch
from .enums import LossReduction

class Reduction:
    def __init__(self, reduction: LossReduction = 'none') -> None:
        self.func = lambda x : x
        if reduction == LossReduction.MEAN:
            self.func = torch.mean 
        elif reduction == LossReduction.SUM:
            self.func = torch.sum
        else:
            self.func = lambda x, **kwargs: x
    
    def __call__(self, input, **kwargs):
        return self.func(input, **kwargs)