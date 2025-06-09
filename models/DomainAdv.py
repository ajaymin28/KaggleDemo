import torch.nn as nn
import torch

# GRL
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

# Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_domains):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
    def forward(self, x):
        return self.head(x)