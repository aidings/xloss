import torch
import torch.nn as nn
from xloss.core import RealFakeLoss, PerceptualLoss, PatchAdversarialLoss

class AEKLLoss(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.disc_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.perceptual_loss = PerceptualLoss('none')
        self.real_fake_loss = RealFakeLoss()
        self.adv_loss = PatchAdversarialLoss() 
 
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):    
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight

        return d_weight
    
    def generator(self, inputs, reconstructions, posteriors, logits_fake, glob_step, last_layer):
        disc_factor = 0 if glob_step < self.disc_iter_start else self.disc_factor
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.mean(nll_loss)

        kl_loss = posteriors.kl() 
        kl_loss = torch.mean(kl_loss)

        # g_loss = -torch.mean(logits_fake)
        g_loss = self.adv_loss(logits_fake, True, False)

        if disc_factor > 0.0:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
        else:
            d_weight = torch.tensor(0.0)

        loss = nll_loss + self.kl_weight * kl_loss + disc_factor * d_weight * g_loss
        loss_dict = {
            "gtotal": loss.item(),
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item(),
            "g_loss": g_loss.item(),
            "p_loss": p_loss.mean().item(),
            "logvar": self.logvar.item()
        }
        return loss, loss_dict
    
    def discriminator(self, logits_real, logits_fake, glob_step):
        disc_factor = 0 if glob_step < self.disc_iter_start else self.disc_factor
        d_loss = disc_factor * self.real_fake_loss(logits_real, logits_fake)
        loss_dict = {
            "dtotal": d_loss.item(),
            "real": logits_real.mean().item(),
            "fake": logits_fake.mean().item()
        }
        return d_loss, loss_dict
    
