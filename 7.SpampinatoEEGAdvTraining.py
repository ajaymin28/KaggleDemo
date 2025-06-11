import torch
import torch.nn as nn
import torchvision.models as models
from utils.EEGDataset import EEGDataset
import argparse
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from utils.transforms import mynet_transform
from utils.argsparsers import getSpampinatoArgs

from models.eeg import EEGTSConv

if __name__=="__main__":

    FLAGS, unparsed = getSpampinatoArgs()
    print(FLAGS)
    
    SUBJECT = FLAGS.subject
    BATCH_SIZE = int(FLAGS.batch_size)
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    EEG_DATASET_PATH = FLAGS.dataset


    ADV_TRAIN = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = EEGTSConv(adv_training=ADV_TRAIN, num_subjects=5).to(device)


    eeg_dataset = EEGDataset(eeg_signals_path=f"{EEG_DATASET_PATH}",
           eeg_splits_path=f"{FLAGS.dataset_split}",
           imagesRoot=f"{FLAGS.imagenet_root}",
           subject=0,
           subset="train",
           exclude_subjects=[6],
           preprocessin_fn=mynet_transform)

    val_eeg_dataset = EEGDataset(eeg_signals_path=f"{EEG_DATASET_PATH}",
            eeg_splits_path=f"{FLAGS.dataset_split}",
            imagesRoot=f"{FLAGS.imagenet_root}",
            subject=0,
            subset="val",
            exclude_subjects=[6],
            preprocessin_fn=mynet_transform)
    
    test_eeg_dataset = EEGDataset(eeg_signals_path=f"{EEG_DATASET_PATH}",
            eeg_splits_path=f"{FLAGS.dataset_split}",
            imagesRoot=f"{FLAGS.imagenet_root}",
            subject=6,
            subset="test",
            exclude_subjects=[6],
            preprocessin_fn=mynet_transform)

    # # Example input image
    # input_image = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image of size 224x224

    dataloader = torch.utils.data.DataLoader(
            eeg_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
            test_eeg_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
            val_eeg_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )

    
    import os
    # os.makedirs(FLAGS.log_dir, exist_ok=True)


    if FLAGS.mode=="train":

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.5, 0.9),weight_decay=1e-4)
        cls_loss_fn = torch.nn.CrossEntropyLoss().to(device)
        domain_loss_fn =  torch.nn.CrossEntropyLoss().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min')

        

        All_Losses = []
        Val_All_losses = []
        Test_All_losses = []


        DOMAIN_LAMBDA = 1.5
        for epoch in range(EPOCHS):
            model.train()

            total_train_loss = 0
            total_val_loss = 0

            acc = {
                "train": {
                    "total_samples" : 0,
                    "total_correct" : 0,
                    "dom_loss": 0,
                    "img_loss": 0,
                    "total_loss": 0
                },
                "val": {
                    "total_samples" : 0,
                    "total_correct" : 0,
                    "dom_loss": 0,
                    "img_loss": 0,
                    "total_loss": 0
                },
                "test": {
                    "total_samples" : 0,
                    "total_correct" : 0
                }
            }
                
            i =0 
            len_dataloader = len(dataloader)
            alpha_smoothing_factor = -5
            subject_id_of_interest = 0  # for example on which cls loss will be calculated

            for i, (eeg, label, image, subject_labels, image_features) in enumerate(dataloader):
                # Prepare inputs and labels
                eeg = eeg.transpose(2, 1).to(device)             # (batch, channel, time)
                cls_labels = label["ClassId"].to(device)
                subject_labels = subject_labels.to(device)
                mask = (subject_labels == subject_id_of_interest)

                # Adversarial alpha scheduling
                p = float(i + epoch * len_dataloader) / (EPOCHS * len_dataloader)
                alpha = 2. / (1. + np.exp(alpha_smoothing_factor * p)) - 1

                # Model forward
                if ADV_TRAIN:
                    cls_logits, domain_logits, features = model(eeg, alpha)
                    if mask.any():
                        img_cls_loss = cls_loss_fn(cls_logits[mask], cls_labels[mask]) if mask.any() else 0.0
                    else:
                        img_cls_loss = 0.0
                    dom_loss = domain_loss_fn(domain_logits, subject_labels) * DOMAIN_LAMBDA
                else:
                    cls_logits,_, features = model(eeg)
                    img_cls_loss = cls_loss_fn(cls_logits, cls_labels)
                    dom_loss = 0.0

                # Total loss
                total_loss = img_cls_loss + dom_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accuracy computation (only on subject of interest)
                if mask.any():
                    preds = cls_logits[mask].argmax(dim=1)
                    correct = (preds == cls_labels[mask]).sum().item()
                    num_samples = cls_logits[mask].size(0)
                else:
                    correct = 0
                    num_samples = 0

                # Logging
                acc["train"]["img_loss"] += float(img_cls_loss)
                acc["train"]["dom_loss"] += float(dom_loss)
                acc["train"]["total_loss"] += float(total_loss)
                acc["train"]["total_correct"] += correct
                acc["train"]["total_samples"] += num_samples


            avg_train_loss = total_train_loss/len(dataloader)

            model.eval()
            with torch.no_grad():
                for veeg, vlabel, vimage, vsubject_labels, vimage_features in val_dataloader:
                    veeg = veeg.transpose(2, 1).to(device)  # (batch, channel, time)
                    vcls_labels = vlabel["ClassId"].to(device)
                    vsubject_labels = vsubject_labels.to(device)
                    vmask = (vsubject_labels == subject_id_of_interest)

                    domain_loss = 0.0

                    if ADV_TRAIN:
                        vcls_logits, vdomain_logits, vfeatures = model(veeg, alpha=0)

                        # Classification loss and accuracy only for subject of interest
                        if vmask.any():
                            val_img_loss = cls_loss_fn(vcls_logits[vmask], vcls_labels[vmask])
                            vpreds = vcls_logits[vmask].argmax(dim=1)
                            vcorrect = (vpreds == vcls_labels[vmask]).sum().item()
                            vsamples = vcls_logits[vmask].size(0)
                        else:
                            val_img_loss = torch.tensor(0.0, device=veeg.device)
                            vcorrect = 0
                            vsamples = 0

                        domain_loss = domain_loss_fn(vdomain_logits, vsubject_labels).item()
                    else:
                        vcls_logits,_, vfeatures = model(veeg)
                        val_img_loss = cls_loss_fn(vcls_logits, vcls_labels)
                        vpreds = vcls_logits.argmax(dim=1)
                        vcorrect = (vpreds == vcls_labels).sum().item()
                        vsamples = vcls_logits.size(0)
                    

                    val_total_loss = val_img_loss + domain_loss

                    acc["val"]["img_loss"] += float(val_img_loss)
                    acc["val"]["dom_loss"] += float(domain_loss)
                    acc["val"]["total_loss"] += float(val_total_loss)
                    acc["val"]["total_correct"] += vcorrect
                    acc["val"]["total_samples"] += vsamples


            avg_train_dom_loss = acc["train"]["dom_loss"]/len(dataloader)
            avg_train_img_loss = acc["train"]["img_loss"]/len(dataloader)
            avg_train_total_loss = acc["train"]["total_loss"]/len(dataloader)
            avg_train_acc = acc["train"]["total_correct"]/acc["train"]["total_samples"]

            avg_val_dom_loss = acc["val"]["dom_loss"]/len(dataloader)
            avg_val_img_loss = acc["val"]["img_loss"]/len(dataloader)
            avg_val_total_loss = acc["val"]["total_loss"]/len(dataloader)
            avg_val_acc = acc["val"]["total_correct"]/acc["val"]["total_samples"]

            scheduler.step(avg_val_total_loss)


            print(f"\n[{epoch+1}/{EPOCHS}][TRAIN] acc:{avg_train_acc:.2f} loss: {avg_train_total_loss:.3f} img loss: {avg_train_img_loss:.3f} dom loss:{avg_train_dom_loss:.3f} alpha:{alpha:.2f} lr: {scheduler.get_last_lr()}")
            print(f"[{epoch+1}/{EPOCHS}][VAL] acc:{avg_val_acc:.2f} loss: {avg_val_total_loss:.3f} img loss: {avg_val_img_loss:.3f} dom loss:{avg_val_dom_loss:.3f}")

        
        # Test 
        acc = {
            "test": {
                    "total_samples" : 0,
                    "total_correct" : 0
            }
        }

        model.eval()
        with torch.no_grad():
            for (veeg, vlabel,vimage,vi, vimage_features) in test_dataloader:
                veeg = veeg.transpose(2,1) # batch, channel, time
                vcls_labels = vlabel["ClassId"].to(device)
                vcls_logits,_, vfeatures = model(veeg.to(device))
                # val_loss = loss_fn(vcls_logits, vcls_labels)
                
                vpreds = vcls_logits.argmax(dim=1)
                vcorrect = (vpreds == vcls_labels).sum().item()

                acc["test"]["total_correct"] += vcorrect
                acc["test"]["total_samples"] += vcls_logits.size(0)
                # total_val_loss += val_loss.item()

            avg_test_acc = acc["test"]["total_correct"]/acc["test"]["total_samples"]
            print(avg_test_acc)

    else:
        # TODO Test only code
        pass