import torch.nn as nn
import torch

# Gradient Reversal Layer (GRL)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self,embed_dim=512, num_domains=2, alpha=1.0):
        super(DomainDiscriminator, self).__init__()

        self.alpha = alpha
        self.num_domains = num_domains

        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_domains),  # Predict subject
            nn.LogSoftmax()
        ) 
        
    def forward(self, x, alpha=None):
        if alpha is None: 
            alpha = self.alpha
            
        reversed_x = grad_reverse(x, alpha)  # Apply GRL
        domain_features = self.feature_extractor(reversed_x)  # Extract subject features
        domain_pred = self.classifier(domain_features)  # Predict subject
        return domain_pred, domain_features  # Return both