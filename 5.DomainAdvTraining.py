from torch.amp import autocast, GradScaler
import numpy as np
from utils.datasets import DomainNetDataset
from utils.transforms import mynet_transform
from torch.utils.data import DataLoader, random_split
from models.models import MyNet
import torch.nn as nn
from torch.optim import Adam
import torch
from utils.config import load_config, parse_args
from utils.utilities import timeit,is_wandb_logged_in
import os
import wandb

def eval_model(model, loader, device, image_cls_loss, domain_loss, alpha=0):
    model.eval()
    img_loss = 0
    dom_loss = 0
    total = 0
    cls_correct = 0
    dom_correct = 0

    with torch.no_grad():
        for data in loader:
            imgs = data["image"].to(device)
            
            cls_labels = data["label"].to(device)
            cls_pred_x, _, domain_cls_pred, _ = model(imgs, alpha)
            iloss = image_cls_loss(cls_pred_x, cls_labels)

            if cfg.adv_training:
                domain_labels = data["domain"].to(device)
                dloss = domain_loss(domain_cls_pred, domain_labels)

            img_loss += iloss.item() * imgs.size(0)
            if cfg.adv_training:
                dom_loss += dloss.item() * imgs.size(0)
            total += imgs.size(0)
            cls_correct += (cls_pred_x.argmax(1) == cls_labels).sum().item()
            if cfg.adv_training:
                dom_correct += (domain_cls_pred.argmax(1) == domain_labels).sum().item()

    return (
        img_loss / total,
        dom_loss / total,
        cls_correct / total,
        dom_correct / total,
    )


@timeit
def train():
    len_dataloader = len(train_loader)

    for epoch in range(cfg.n_epochs):
        model.train()

        train_img_loss = 0
        train_dom_loss = 0
        train_total = 0
        train_cls_correct = 0
        train_dom_correct = 0

        for i, data in enumerate(train_loader):
            t_img = data["image"].to(device)  
            t_cls_labels = data["label"].to(device)

            alpha = cfg.adv_alpha
            if cfg.schedule_alpha:
                p = float(i + epoch * len_dataloader) / (cfg.n_epochs * len_dataloader)
                alpha = 2. / (1. + np.exp(cfg.alpha_smoothing_factor * p)) - 1

            if cfg.use_fp16:
                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    cls_pred_x, base_feat, domain_cls_pred, domain_features = model(t_img, alpha)
                    img_loss = image_cls_loss(cls_pred_x, t_cls_labels)
                    total_loss = img_loss
                    if cfg.adv_training:
                        t_domain_labels = data["domain"].to(device)
                        dom_loss = domain_loss(domain_cls_pred, t_domain_labels)
                        total_loss = img_loss + dom_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_pred_x, base_feat, domain_cls_pred, domain_features = model(t_img, alpha)
                img_loss = image_cls_loss(cls_pred_x, t_cls_labels)
                total_loss = img_loss
                if cfg.adv_training:
                    t_domain_labels = data["domain"].to(device)
                    dom_loss = domain_loss(domain_cls_pred, t_domain_labels)
                    total_loss += dom_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_img_loss += img_loss.item() * t_img.size(0)
            if cfg.adv_training:
                train_dom_loss += dom_loss.item() * t_img.size(0)

            train_total += t_img.size(0)
            train_cls_correct += (cls_pred_x.argmax(1) == t_cls_labels).sum().item()
            if cfg.adv_training: 
                train_dom_correct += (domain_cls_pred.argmax(1) == t_domain_labels).sum().item()

        avg_train_img_loss = train_img_loss / train_total
        
        if cfg.adv_training: 
            avg_train_dom_loss = train_dom_loss / train_total
        train_cls_acc = train_cls_correct / train_total
        if cfg.adv_training: 
            train_dom_acc = train_dom_correct / train_total

        log_dict = {
            "epoch": epoch + 1,
            "train/img_loss": avg_train_img_loss,
            "train/cls_acc": train_cls_acc,
        }

        if cfg.adv_training:
            log_dict.update({
                "train/dom_loss": avg_train_dom_loss,
                "train/dom_acc": train_dom_acc,
                "train_alpha": alpha
            })
            msg = (
                f"Epoch {epoch+1}: "
                f"Train ImgLoss: {avg_train_img_loss:.4f}, DomLoss: {avg_train_dom_loss:.4f}, "
                f"ClsAcc: {train_cls_acc:.3f}, DomAcc: {train_dom_acc:.3f} Alpha: {alpha}"
            )
        else:
            msg = (
                f"Epoch {epoch+1}: "
                f"Train ImgLoss: {avg_train_img_loss:.4f},  "
                f"ClsAcc: {train_cls_acc:.3f}"
            )

        if epoch % 2 == 0:
            val_results = eval_model(model, val_loader, device, image_cls_loss, domain_loss, alpha=0)
            test_results = eval_model(model, test_loader, device, image_cls_loss, domain_loss, alpha=0)

            log_dict.update({
                "val/img_loss": val_results[0],
                "val/cls_acc": val_results[2],
                "test/img_loss": test_results[0],
                "test/cls_acc": test_results[2],
            })

            if cfg.adv_training:

                log_dict.update({
                    "val/dom_loss": val_results[1],
                    "val/dom_acc": val_results[3],
                    "test/dom_loss": test_results[1],
                    "test/dom_acc": test_results[3],
                })

                msg += (
                    f" | Val ImgLoss: {val_results[0]:.4f}, DomLoss: {val_results[1]:.4f}, "
                    f"ClsAcc: {val_results[2]:.3f}, DomAcc: {val_results[3]:.3f}"
                    f" | Test ImgLoss: {test_results[0]:.4f}, DomLoss: {test_results[1]:.4f}, "
                    f"ClsAcc: {test_results[2]:.3f}, DomAcc: {test_results[3]:.3f}"
                )
            else:
                msg += (
                    f" | Val ImgLoss: {val_results[0]:.4f},  "
                    f"ClsAcc: {val_results[2]:.3f} "
                    f" | Test ImgLoss: {test_results[0]:.4f}, "
                    f"ClsAcc: {test_results[2]:.3f}"
                )
        
        if isWandbLoggedIn:
            wandb.log(log_dict)

        print(msg)


if __name__=="__main__":
    
    args = parse_args()
    cfg = load_config(args)
    print(cfg)

    f_labels = cfg.domainnet_classes[:cfg.n_classes]
    print(f_labels)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key is None:
        print("WANDB_API_KEY not found in Kaggle Secrets.")
        if cfg.wandb_apikey!="":
            wandb.login(key=cfg.wandb_apikey)
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

        wandb.init(
            project=cfg.wandb_project,          # your project name
            config=cfg.model_dump()             # log all config parameters
        )
    

    dataset_train = DomainNetDataset(
        root_dir=cfg.data_path,
        domains=cfg.TRAIN_DOMAINS,
        split="train",
        transform=mynet_transform,
        classes=f_labels
    )

    val_ratio = 0.2
    n_train = int(len(dataset_train) * (1 - val_ratio))
    n_val = len(dataset_train) - n_train
    train_dataset, val_dataset = random_split(dataset_train, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # for a specific set of classes only
    dataset_test = DomainNetDataset(
        root_dir=cfg.data_path,
        domains=cfg.TEST_DOMAINS,
        split="test",
        classes=f_labels,
        transform=mynet_transform,
    )
    test_loader  = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_loss = nn.NLLLoss().to(device)
    image_cls_loss = nn.NLLLoss().to(device)

    model = MyNet(cfg).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler()
    train()