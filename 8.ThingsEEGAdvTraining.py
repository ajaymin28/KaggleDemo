import torch
# import torch.nn as nn
import numpy as np
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import os
# from utils.transforms import mynet_transform
from utils.EEGThingsDataset import EEGThingsDataset
from utils.common import TrainConfig
import torch
from utils.losses import MMDLoss, ContrastiveLoss, SupervisedContrastiveLoss
from utils.utilities import is_wandb_logged_in, config_to_dict
from utils.RunManager import RunManager, CheckpointManager
from models.eeg import ThingsEEGConv
import wandb
import os
import pandas as pd

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_train_config_parser():
    parser = argparse.ArgumentParser(description="TrainConfig Arguments")

    parser.add_argument("--local_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=63)
    parser.add_argument("--time_points", type=int, default=250)
    parser.add_argument("--sessions", type=int, default=4)

    parser.add_argument("--mean_eeg_data", type=str2bool, default=False)
    parser.add_argument("--keep_dim_after_mean", type=str2bool, default=False)
    parser.add_argument("--cache_data", type=str2bool, default=True)

    parser.add_argument("--image_feature_dim", type=int, default=768)
    parser.add_argument("--num_subjects", type=int, default=1)
    parser.add_argument("--load_pre_trained_models", type=str2bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--discriminator_lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--eeg_data_path", type=str, default=None)  # Default: global_config.EEG_DATA_PATH
    parser.add_argument("--test_center_path", type=str, default=None)  # Default: global_config.TEST_CENTER_PATH
    parser.add_argument("--model_save_base_dir", type=str, default=None)  # Default: global_config.MODEL_BASE_DIR
    parser.add_argument("--data_base_dir", type=str, default=None)  # Default: global_config.DATA_BASE_DIR

    parser.add_argument("--Contrastive_augmentation", type=str2bool, default=True)
    parser.add_argument("--nSub", type=int, default=1)
    parser.add_argument("--nSub_Contrastive", type=int, default=2)
    parser.add_argument("--EEG_Augmentation", type=str2bool, default=False)
    parser.add_argument("--Total_Subjects", type=int, default=2)
    # parser.add_argument("--MultiSubject", type=str2bool, default=True)
    parser.add_argument("--TestSubject", type=int, default=1)
    parser.add_argument("--dnn", type=str, default="clip")

    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--max_lambda_adv", type=float, default=2.0)
    parser.add_argument("--enable_adv_training", type=str2bool, default=False)
    parser.add_argument("--lambda_ortho", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.0)

    parser.add_argument("--log_test_data", type=str2bool, default=True)
    parser.add_argument("--Train", type=str2bool, default=True)
    parser.add_argument("--profile_code", type=str2bool, default=False)
    parser.add_argument("--encoder_output_dim", type=int, default=768)
    parser.add_argument("--wandb_apikey", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="things_eeg_adv_training")

    parser.add_argument("--NUM_CLASSES", type=int, default=1654)
    parser.add_argument("--TRAIN_SUBJECT_IDS", nargs="+", type=int, default=[1],
                        help="Space separated subject IDs for training, e.g. --TRAIN_SUBJECT_IDS 1 2 3")
    parser.add_argument("--VALIDATION_SUBJECT_IDS", nargs="+", type=int, default=[1],
                        help="Space separated subject IDs for validation")
    parser.add_argument("--TRAIN_SESSION_IDS", nargs="+", type=int, default=[0,1],
                        help="Space separated session IDs for training")
    parser.add_argument("--VALIDATION_SESSION_IDS", nargs="+", type=int, default=[2],
                        help="Space separated session IDs for validation")
    parser.add_argument("--ONE_SUBJECT_CLS", type=str2bool, default=True,
                        help="Set True to use only one subject's classification loss (as per DANN paper)")



    return parser

# Example usage:
# parser = get_train_config_parser()
# args = parser.parse_args()
# print(args)


if __name__=="__main__":

    ChangeNotes = """
        [06/11/2025][05:05PM]: Used session 0,1 of subject 1 to train and session 2 as validation, session samples are used seperately instead of mean.
        [06/11/2025][05:05PM]: Changed time samples to 32 after conv1d.
    """

    parser = get_train_config_parser()
    cli_args = parser.parse_args()


    args = TrainConfig()

    args.batch_size = 512
    args.enable_adv_training = False
    EPOCHS = args.local_epochs = 100
    args.lambda_adv = 1.2
    args.num_subjects = 1
    
    args.learning_rate = 0.0002
    args.keep_dim_after_mean = False
    args.mean_eeg_data = False # mean of sessions will be done (NICE does it), if false sessions will be seperated and each session will be treated as a seperate sample

    NUM_CLASSES = 1654
    TRAIN_SUBJECT_IDS = [1]
    VALIDATION_SUBJECT_IDS = [1]
    TRAIN_SESSION_IDS = [0,1]  # upto 4 sessions for train and val set
    VALIDATION_SESSION_IDS = [2] # different sessions are used for train and val, dont use same as train since we are not splitting data from same sessions.
    ONE_SUBJECT_CLS = True # only one subject's cls loss will be done (as per DANN paper), not applicable for Non Adv Training.

    # override CLI argument values
    for key, value in vars(cli_args).items():
        # Only set the attribute if it exists in TrainConfig, optional safety:
        if hasattr(args, key):
            print(f"setting: {key} -> {value}")
            setattr(args, key, value)


    if args.enable_adv_training:
        # adv training possible when subjects are more than one
        assert args.num_subjects>=2 and len(set(TRAIN_SUBJECT_IDS))>=2

    try:
        wandb.finish() # close existing run
    except Exception:
        pass


    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key is None:
        print("WANDB_API_KEY not found in Kaggle Secrets.")
        if args.wandb_apikey!="":
            wandb.login(key=args.wandb_apikey)
        else:
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
                wandb.login(key=wandb_api_key)
            except:
                print("WANDB_API_KEY not found in config/args/kaggle sec")
    else:
        wandb.login(key=wandb_api_key)

    isWandbLoggedIn = is_wandb_logged_in()
    if isWandbLoggedIn:
        print(f"wandb logged in...")
        # config_dict = {
        #     k: v for k, v in TrainConfig.__dict__.items()
        #     if not k.startswith('__') and not callable(getattr(TrainConfig, k))
        # }
        dict_args = config_to_dict(args)
        wandb.init(
            project="EEGDomainADV",          # your project name
            # mode="offline"
            config=dict_args,             # log all config parameters
            notes=ChangeNotes
        )

    train_ds = EEGThingsDataset(args=args,nsubs=TRAIN_SUBJECT_IDS, session_ids=TRAIN_SESSION_IDS,subset="train")
    val_ds = EEGThingsDataset(args=args,nsubs=VALIDATION_SUBJECT_IDS, session_ids=VALIDATION_SESSION_IDS,subset="val")

    dataloader = torch.utils.data.DataLoader(train_ds,batch_size=args.batch_size,num_workers=0,shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=args.batch_size,num_workers=0,shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThingsEEGConv(adv_training=args.enable_adv_training, num_subjects=args.num_subjects, n_classes=NUM_CLASSES).to(device)

    runManager = RunManager(run_id=wandb.run.id)
    cpm_ThingsEEG = CheckpointManager(prefix="Thingseeg",base_dir=f"./checkpoints/{runManager.getRunID()}")


    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    # loss_fn = ContrastiveLoss(temperature=0.07).to(device)
    domain_loss_fn =  torch.nn.CrossEntropyLoss().to(device)
    # domain_loss_fn = MMDLoss().to(device)

    
    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0
        total_val_loss = 0

        acc = {
            "train": {
                "total_samples" : 1,
                "total_correct" : 0,
                "dom_loss": 0,
                "img_loss": 0,
                "total_loss": 0
            },
            "val": {
                "total_samples" : 1,
                "total_correct" : 0,
                "dom_loss": 0,
                "img_loss": 0,
                "total_loss": 0
            },
            "test": {
                "total_samples" : 1,
                "total_correct" : 0
            }
        }
            
        

        i =0 
        len_dataloader = len(dataloader)
        alpha_smoothing_factor = -5
        subject_id_of_interest = 0  # on which cls loss will be calculated

        for i, (eeg, label,image_feat, subject_labels) in enumerate(dataloader):
            # Prepare inputs and labels
            eeg = eeg.type(torch.cuda.FloatTensor).to(device, non_blocking=True) 
            # eeg = eeg.transpose(2, 1).to(device, non_blocking=True)           # (batch, channel, time)
            cls_labels = label.to(device, non_blocking=True)
            subject_labels = subject_labels.to(device, non_blocking=True)
            mask = (subject_labels == subject_id_of_interest)
            image_feat = image_feat.to(device, non_blocking=True)

            # Adversarial alpha scheduling
            p = float(i + epoch * len_dataloader) / (EPOCHS * len_dataloader)
            alpha = 2. / (1. + np.exp(alpha_smoothing_factor * p)) - 1

            dom_loss = 0.0

            # Model forward
            if args.enable_adv_training:
                cls_logits, domain_logits, features = model(eeg, alpha)

                if ONE_SUBJECT_CLS:
                    if mask.any():
                        # img_cls_loss = loss_fn(cls_logits[mask], cls_labels[mask]) if mask.any() else 0.0
                        # img_cls_loss = loss_fn(features[mask], image_feat[mask]) if mask.any() else 0.0
                        img_cls_loss = model.nice_contrastive_loss(features[mask], image_feat[mask])
                    else:
                        img_cls_loss = 0.0
                else:
                    # img_cls_loss = loss_fn(features, image_feat)
                    img_cls_loss = model.nice_contrastive_loss(features, image_feat)


                dom_loss = domain_loss_fn(domain_logits, subject_labels) * args.lambda_adv
            else:
                # print("input shape before model call", eeg.shape)
                cls_logits,_, features = model(eeg)
                # img_cls_loss = loss_fn(cls_logits, cls_labels)
                # img_cls_loss = loss_fn(features, image_feat)
                img_cls_loss = model.nice_contrastive_loss(features, image_feat)
                

            # Total loss
            total_loss = img_cls_loss + dom_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

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
        for veeg, vlabel,vimage_feat, vsubject_labels in val_dataloader:
            veeg = veeg.to(device, non_blocking=True).type(torch.cuda.FloatTensor)    # (batch, channel, time)
            vcls_labels = vlabel.to(device, non_blocking=True)
            vsubject_labels = vsubject_labels.to(device, non_blocking=True)
            vmask = (vsubject_labels == subject_id_of_interest)
            vimage_feat = vimage_feat.to(device, non_blocking=True)

            domain_loss = 0.0
            vcorrect = 0
            vsamples = 0

            if args.enable_adv_training:
                vcls_logits, vdomain_logits, vfeatures = model(veeg, alpha=0)

                if ONE_SUBJECT_CLS:
                    # Classification loss and accuracy only for subject of interest
                    if vmask.any():
                        # val_img_loss = loss_fn(vcls_logits[vmask], vcls_labels[vmask])
                        # val_img_loss = loss_fn(vfeatures[vmask], vimage_feat[vmask])
                        val_img_loss = model.nice_contrastive_loss(vfeatures[vmask], vimage_feat[vmask])
                    else:
                        val_img_loss = torch.tensor(0.0, device=veeg.device)
                else:
                    # val_img_loss = loss_fn(vfeatures, vimage_feat)
                    val_img_loss = model.nice_contrastive_loss(vfeatures, vimage_feat)

                domain_loss = domain_loss_fn(vdomain_logits, vsubject_labels).item()

            else:
                vcls_logits,_, vfeatures = model(veeg)
                # val_img_loss = loss_fn(vcls_logits, vcls_labels)
                # val_img_loss = loss_fn(vfeatures, vimage_feat)
                val_img_loss = model.nice_contrastive_loss(vfeatures, vimage_feat)

            val_total_loss = val_img_loss + domain_loss

            acc["val"]["img_loss"] += float(val_img_loss)
            acc["val"]["dom_loss"] += float(domain_loss)
            acc["val"]["total_loss"] += float(val_total_loss)
            acc["val"]["total_correct"] += vcorrect
            acc["val"]["total_samples"] += vsamples

            # total_val_loss += val_loss.item()

        # avg_val_loss = total_val_loss/len(dataloader)

        avg_train_dom_loss = acc["train"]["dom_loss"]/len(dataloader)
        avg_train_img_loss = acc["train"]["img_loss"]/len(dataloader)
        avg_train_total_loss = acc["train"]["total_loss"]/len(dataloader)
        avg_train_acc = acc["train"]["total_correct"]/acc["train"]["total_samples"]

        avg_val_dom_loss = acc["val"]["dom_loss"]/len(dataloader)
        avg_val_img_loss = acc["val"]["img_loss"]/len(dataloader)
        avg_val_total_loss = acc["val"]["total_loss"]/len(dataloader)
        avg_val_acc = acc["val"]["total_correct"]/acc["val"]["total_samples"]


        print(f"\n[{epoch+1}/{EPOCHS}][TRAIN] acc:{avg_train_acc:.2f} loss: {avg_train_total_loss:.2f} cls loss: {avg_train_img_loss:.3f} dom loss:{avg_train_dom_loss:.3f} alpha:{alpha:.2f}")
        print(f"[{epoch+1}/{EPOCHS}][VAL] acc:{avg_val_acc:.2f} loss: {avg_val_total_loss:.2f} cls loss: {avg_val_img_loss:.3f} dom loss:{avg_val_dom_loss:.3f}")

        data = {
            "train/acc": avg_train_acc,
            "train/total_loss": round(avg_train_total_loss,6),
            "train/cls_loss": round(avg_train_img_loss,6),
            "train/dom_loss": avg_train_dom_loss,
            "train/alpha": alpha,
            "val/acc": avg_val_acc,
            "val/total_loss": round(avg_val_total_loss,6),
            "val/cls_loss": round(avg_val_img_loss,6),
            "val/dom_loss": avg_val_dom_loss,
            "epoch": epoch + 1,
        }
        # print(data)
        try:
            wandb.log(data)
        except Exception as e:
            print(f"error logging the data :{e}")



    # TEST
    test_center = np.load(args.test_center_path + '\\center_' + args.dnn + '.npy', allow_pickle=True)
    test_center = torch.from_numpy(test_center)
    test_center = test_center.to(device, non_blocking=True).type(torch.cuda.FloatTensor)
    args.nSub = 3

    

    for sub_i in range(1,5):
        args.nSub = sub_i

        results_ses_avg = {}

        for session_i in range(80):

            test_eeg, test_label = EEGThingsDataset.get_eeg_data(args, sessions=[session_i])

            test_eeg = torch.from_numpy(test_eeg)
            # test_img_feature = torch.from_numpy(test_img_feature)
            test_label = torch.from_numpy(test_label)
            test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

            total = 0
            top1 = 0
            top3 = 0
            top5 = 0
            
            with torch.no_grad():
                for i, (teeg, tlabel) in enumerate(test_dataloader):

                    teeg = teeg.to(device, non_blocking=True).type(torch.cuda.FloatTensor)
                    tlabel = tlabel.to(device, non_blocking=True).type(torch.cuda.LongTensor)

                    teeg = teeg.to(device, non_blocking=True).type(torch.cuda.FloatTensor)

                    _,_, tfeatures = model(teeg)
                    tfea = tfeatures / tfeatures.norm(dim=1, keepdim=True)


                    similarity = (100.0 * tfea @ test_center.t()).softmax(dim=-1)  # no use 100?
                    _, indices = similarity.topk(5)

                    tt_label = tlabel.view(-1, 1)
                    total += tlabel.size(0)
                    top1 += (tt_label == indices[:, :1]).sum().item()
                    top3 += (tt_label == indices[:, :3]).sum().item()
                    top5 += (tt_label == indices).sum().item()

                    
                top1_acc = float(top1) / float(total)
                top3_acc = float(top3) / float(total)
                top5_acc = float(top5) / float(total)

            print(f"Subject: {args.nSub} Session {session_i} | Top1: {top1_acc:.3f}, Top3: {top3_acc:.3f}, Top5: {top5_acc:.3f}")

            # In your loop:

            for tk in [1,3,5]:
                if f"Subject_{args.nSub}/Top{tk}" not in results_ses_avg.keys():
                    results_ses_avg[f"Subject_{args.nSub}/Top{tk}"] = []

            # if "sessions" not in results.keys():
                # results["sessions"] =  []


            # if f"{session_i}" not in results.keys():
            #     results[f"{args.nSub}"][f"{session_i}"] = {}

            results_ses_avg[f"Subject_{args.nSub}/Top1"].append(top1_acc)
            results_ses_avg[f"Subject_{args.nSub}/Top3"].append(top3_acc)
            results_ses_avg[f"Subject_{args.nSub}/Top5"].append(top5_acc)
            # results["sessions"].append(session_i)

            results = {
                f"Subject_{args.nSub}/Top1": top1_acc,
                f"Subject_{args.nSub}/Top3": top3_acc,
                f"Subject_{args.nSub}/Top5": top5_acc,
                "session": session_i +1
            }
            try:
                wandb.log(results)
            except Exception as e:
                print(f"error logging the test data :{e}")

        results_ses_avg[f"Subject_{args.nSub}/Top1"] = np.array(results_ses_avg[f"Subject_{args.nSub}/Top1"]).mean()
        results_ses_avg[f"Subject_{args.nSub}/Top3"] = np.array(results_ses_avg[f"Subject_{args.nSub}/Top3"]).mean()
        results_ses_avg[f"Subject_{args.nSub}/Top5"] = np.array(results_ses_avg[f"Subject_{args.nSub}/Top5"]).mean()

        results = {
            f"Subject_{args.nSub}/Top1_avg": top1_acc,
            f"Subject_{args.nSub}/Top3_avg": top3_acc,
            f"Subject_{args.nSub}/Top5_avg": top5_acc,
        }

        try:
            wandb.log(results)
        except Exception as e:
            print(f"error logging the test data :{e}")


    # TODO DO THIS when VAL is best
    cpm_ThingsEEG.save_checkpoint(model=model,optimizer=optimizer,epoch=epoch)

    try:
        wandb.finish()
    except Exception as e:
        print(e)




    # if isWandbLoggedIn:
    #     df = pd.DataFrame(results)
    #     per_subject_avg = df.groupby("Subject")[["Top1", "Top3", "Top5"]].mean().reset_index()
    #     per_subject_avg["Session"] = -1  # Use -1 or any placeholder to denote 'mean'

    #     # If you want to distinguish these rows, add a flag or label
    #     per_subject_avg["Averaged"] = True
    #     df["Averaged"] = False

    #     # Combine
    #     df_with_subject_avg = pd.concat([df, per_subject_avg], ignore_index=True)

    #     table = wandb.Table(dataframe=df)

    #     # Log the extended table
    #     table = wandb.Table(dataframe=df_with_subject_avg)  # or df_with_subject_avg
    #     wandb.log({"Test Metrics": table})