import torch.nn as nn
import torch
from models.DomainAdv import DomainDiscriminator
from utils.config import ModelConfig


DEBUG = ModelConfig.DEBUG

class MyNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MyNet, self).__init__()

        self.config = config
        n_classes = config.n_classes
        image_size = config.image_size
        num_domains = config.num_domains
        self.adv_training = config.adv_training

        self.conv_model = nn.Sequential(
            nn.Conv2d(3,16,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(16,32,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(32,8,(3,3), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
        )

        dummy_input = torch.rand(size=(1,3,image_size,image_size), dtype=torch.float32)
        conv_out = self.getConv(dummy_input)
        flat_out = nn.Flatten()(conv_out)
        # print(flat_out.size(0))

        self.features_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_out.size(-1), 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, n_classes)
        )
        self.domain_desc = DomainDiscriminator(embed_dim=512,num_domains=num_domains)

        del dummy_input, conv_out, flat_out

    
    def residual(self, x):
        pass

    def getheadFeatures(self, x):
        head_f = self.features_head(x)
        if DEBUG: print(f"head_f shape: {head_f.shape}")
        return head_f

    
    def getClassification(self, x):
        cls_h = self.cls_head(x)
        if DEBUG: print(f"cls_h shape: {cls_h.shape}")
        return cls_h
    
    def getConv(self, x):
        conv_out = self.conv_model(x)
        if DEBUG: print(f"conv_out shape: {conv_out.shape}")
        return self.conv_model(x)

    
    def forward(self, x,alpha=1):
        if DEBUG: print(f"input shape: {x.shape}")
        domain_cls_pred, domain_features = [],[]

        x = self.getConv(x)
        base_feat = self.getheadFeatures(x)
        cls_pred_x = self.getClassification(base_feat)

        if self.adv_training:
            domain_cls_pred, domain_features = self.domain_desc(base_feat, alpha)

        return cls_pred_x,base_feat,domain_cls_pred,domain_features