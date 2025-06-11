import torch
import torch.nn as nn
import numpy as np
from models.DomainAdv import DomainDiscriminator, grad_reverse

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class EEGTSConv(nn.Module):
    def __init__(self, channels=128, time=460, n_classes=40, proj_dim=768,drop_proj=0.5, adv_training=False, num_subjects=2):
        super().__init__()

        self.adv_training = adv_training
        self.num_subjects = num_subjects

        self.channel_conv = nn.Sequential(
            nn.Conv1d(channels,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
            nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU()
        )

        # self.time_conv = nn.Sequential(
        #     nn.Conv1d(time,128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),

        #     nn.Conv1d(128,64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),

        #     nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),
        # )

        self.time_conv = nn.Sequential(
            nn.Conv1d(time,400, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(400),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),

            nn.Conv1d(400,320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(320),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),

            nn.Conv1d(320,160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
        )

        
        dummy_input = torch.rand((1,channels,time))

        channel_conv = self.channel_conv(dummy_input)
        conv_batch, conv_channels, conv_time = channel_conv.shape
        channel_conv = channel_conv.view(conv_batch, conv_time, conv_channels)
        time_conv = self.time_conv(channel_conv)
        time_conv = time_conv.transpose(2,1)
        flatten_feat = time_conv.reshape(1, -1)
        flatten_feat_b, flatten_feat_length = flatten_feat.shape


        self.feature_head = nn.Sequential(
            nn.Linear(flatten_feat_length,proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )), 
        )

        self.cls_head = nn.Sequential(
            nn.Linear(proj_dim,proj_dim//2),
            nn.BatchNorm1d(proj_dim//2),
            nn.GELU(),
            nn.Linear(proj_dim//2, n_classes),
        )


        if self.adv_training:
            self.domain_desc = DomainDiscriminator(in_dim=proj_dim,num_domains=self.num_subjects)


    def forward(self, x, alpha=0.5):
        """
        batch, channels, time
        """

        channel_conv = self.channel_conv(x)
        # print("channel conv", channel_conv.shape)
        conv_batch, conv_channels, conv_time = channel_conv.shape
        channel_conv = channel_conv.view(conv_batch, conv_time, conv_channels)
        time_conv = self.time_conv(channel_conv)
        # print("time conv ", time_conv.shape)

        time_conv = time_conv.transpose(2,1)
        
        features = self.feature_head(time_conv.reshape(conv_batch, -1))
        # print("features ", features.shape)
        cls_logits = self.cls_head(features)

        if self.adv_training:
            cls_domain_logits = self.domain_desc(grad_reverse(features,alpha))
            # cls_logits = self.cls_head(features)
            return cls_logits, cls_domain_logits, features

        return cls_logits,torch.zeros(0), features
        # return cls_logits,torch.zeros(1), features


class ThingsEEGConv(nn.Module):
    def __init__(self, channels=63, time=250, n_classes=1654, proj_dim=768,drop_proj=0.5, adv_training=False, num_subjects=2):
        super().__init__()

        self.adv_training = adv_training
        self.num_subjects = num_subjects

        self.channel_conv = nn.Sequential(
            nn.Conv1d(channels,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
            nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU()
        )

        # self.channel_conv = nn.Sequential(
        #     nn.Conv1d(channels,128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),
        #     nn.Conv1d(128,63, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(63),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU()
        # )

        # self.time_conv = nn.Sequential(
        #     nn.Conv1d(time,200, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(200),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),

        #     nn.Conv1d(200,150, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(150),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),

        #     nn.Conv1d(150,100, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(100),
        #     nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
        #     nn.ELU(),
        # )

        self.time_conv = nn.Sequential(
            nn.Conv1d(time,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
            nn.Conv1d(128,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
            nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.ELU(),
        )

        
        dummy_input = torch.rand((1,channels,time))
        channel_conv = self.channel_conv(dummy_input)
        conv_batch, conv_channels, conv_time = channel_conv.shape
        channel_conv = channel_conv.view(conv_batch, conv_time, conv_channels)
        time_conv = self.time_conv(channel_conv)
        time_conv = time_conv.transpose(2,1)
        flatten_feat = time_conv.reshape(1, -1)
        flatten_feat_b, flatten_feat_length = flatten_feat.shape


        self.feature_head = nn.Sequential(
            nn.Linear(flatten_feat_length,proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )), 
        )

        self.cls_head = nn.Sequential(
            nn.Linear(proj_dim,proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, n_classes),
        )


        if self.adv_training:
            self.domain_desc = DomainDiscriminator(in_dim=proj_dim,num_domains=self.num_subjects)


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()


    def nice_contrastive_loss(self, eeg_features, img_features):
        vlabels = torch.arange(eeg_features.shape[0])
        vlabels = vlabels.cuda().type(self.LongTensor)

        eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_eeg = logit_scale * eeg_features @ img_features.t()
        logits_per_img = logits_per_eeg.t()

        loss_eeg = self.criterion_cls(logits_per_eeg, vlabels)
        loss_img = self.criterion_cls(logits_per_img, vlabels)
        loss = (loss_eeg + loss_img) / 2

        return loss

    
    def forward(self, x, alpha=0.5):
        """
        batch, channels, time
        """

        # print("input ", x.shape)
        channel_conv = self.channel_conv(x)
        # print("channel conv", channel_conv.shape)
        conv_batch, conv_channels, conv_time = channel_conv.shape
        channel_conv = channel_conv.view(conv_batch, conv_time, conv_channels)
        time_conv = self.time_conv(channel_conv)
        # print("time conv ", time_conv.shape)

        time_conv = time_conv.transpose(2,1)

        features = self.feature_head(time_conv.reshape(conv_batch, -1))
        # print("features ", features.shape)
        cls_logits = self.cls_head(features)

        if self.adv_training:
            cls_domain_logits = self.domain_desc(grad_reverse(features,alpha))
            # cls_logits = self.cls_head(features)
            return cls_logits, cls_domain_logits, features
        # else:
            
        # print("cls_logits ", cls_logits.shape)

        # return cls_logits, features
        return cls_logits,torch.zeros(1), features