import torch
import torch.nn as nn
import torchvision.models as models
from utils.config import ModelConfig

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


class PretrainedResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, trainable_layers=2):
        super().__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        model = getattr(models, backbone)(pretrained=pretrained)
        # Remove the FC layer, keep everything else
        self.features = nn.Sequential(*(list(model.children())[:-1]))
        self.out_dim = model.fc.in_features

        # Optionally freeze early layers for efficiency/stability
        layers = [self.features[i] for i in range(len(self.features))]
        # You can fine-tune only the last `trainable_layers` blocks
        for layer in layers[:-trainable_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)         # [B, out_dim, 1, 1]
        x = x.view(x.size(0), -1)    # [B, out_dim]
        return x



class DomainAdaptModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.method = config.domain_method
        n_classes, img_size, num_domains = config.n_classes, config.image_size, config.num_domains

        self.encoder = PretrainedResNetEncoder(backbone='resnet18', pretrained=True, trainable_layers=2)
        self.feat_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(512, config.n_classes)

        # Domain head logic remains the same as before
        if self.method == "cdan":
            self.domain_disc = DomainDiscriminator(512 * config.n_classes, config.num_domains)
        else:
            self.domain_disc = DomainDiscriminator(512, config.num_domains)

    def forward(self, x, alpha=1.0, class_prob=None):
        feat = self.encoder(x)
        feat = self.feat_head(feat)
        class_logits = self.classifier(feat)

        if self.method == "cdan":
            if class_prob is None:
                class_prob = torch.softmax(class_logits, dim=1)
                fused = torch.bmm(class_prob.unsqueeze(2), feat.unsqueeze(1)).view(x.size(0), -1)
                domain_input = fused
        else:
            domain_input = feat

        if self.method in ["dann", "cdan"]:
            domain_pred = self.domain_disc(grad_reverse(domain_input, alpha))
        else:
            domain_pred = None

        return class_logits, feat, domain_pred
