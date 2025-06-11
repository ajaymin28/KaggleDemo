import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_features, clip_features, img_classes):
        pred_norm = F.normalize(pred_features, dim=1)
        clip_norm = F.normalize(clip_features, dim=1)
        # Similarity matrix
        logits = torch.matmul(pred_norm, clip_norm.T) / self.temperature

        # Mask where [i, j] = 1 if class[i] == class[j], else 0
        labels = img_classes.view(-1, 1)
        mask = (labels == labels.T).float().to(pred_features.device)
        # Remove self-contrast (don't use anchor itself as positive)
        self_mask = torch.eye(mask.size(0), device=pred_features.device)
        mask = mask - self_mask

        # Numerator: sum of exp(similarity) for positives
        exp_logits = torch.exp(logits)
        numerator = (exp_logits * mask).sum(1)
        # Denominator: sum over all except self
        denominator = (exp_logits * (1 - self_mask)).sum(1)

        # Avoid division by zero
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_features, clip_features):
        # Normalize features
        pred_norm = F.normalize(pred_features, dim=1)
        clip_norm = F.normalize(clip_features, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(pred_norm, clip_norm.T) / self.temperature

        # Targets: each sample matches itself (diagonal of matrix)
        labels = torch.arange(pred_features.size(0), device=pred_features.device)

        # Contrastive loss (InfoNCE style)
        loss = F.cross_entropy(logits, labels)
        return loss

class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # MMD Loss
    @staticmethod
    def gaussian_kernel(self,x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        diff = x - y
        return torch.exp(- (diff ** 2).sum(2) / (2 * sigma ** 2))

    @staticmethod
    def mmd_loss(self,source, target, sigma=1.0):
        K_ss = MMDLoss.gaussian_kernel(source, source, sigma).mean()
        K_tt = MMDLoss.gaussian_kernel(target, target, sigma).mean()
        K_st = MMDLoss.gaussian_kernel(source, target, sigma).mean()
        return K_ss + K_tt - 2 * K_st

    def forward(self, base_features, domain_labels):
        dom_total = domain_labels.size(0)
        dom_loss = torch.tensor(0.0, device=domain_labels.device, requires_grad=True)

        unique_domains = domain_labels.unique()
        mmd_batch = 0
        count = 0
        for i, dom_a in enumerate(unique_domains):
            for dom_b in unique_domains[i+1:]:
                idx_a = (domain_labels == dom_a)
                idx_b = (domain_labels == dom_b)
                if idx_a.sum() > 0 and idx_b.sum() > 0:
                    mmd_batch += MMDLoss.mmd_loss(base_features[idx_a], base_features[idx_b])
                    count += 1
        if count > 0:
            mmd_batch = mmd_batch / count
        else:
            print(f"returning zero loss for mmd: unique domains: {unique_domains}")
            mmd_batch = torch.tensor(0.0, device=domain_labels.device, requires_grad=True)
        dom_loss = mmd_batch * dom_total

        return dom_loss