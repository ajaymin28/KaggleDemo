from torch.amp import autocast, GradScaler
import numpy as np
from utils.datasets import DomainNetDataset
from utils.transforms import mynet_transform
from torch.utils.data import DataLoader, random_split
from models.MMD import DomainAdaptModel, mmd_loss
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
    mmd_loss_total = 0
    total = 0
    cls_correct = 0
    dom_correct = 0

    with torch.no_grad():
        for data in loader:
            imgs = data["image"].to(device)
            cls_labels = data["label"].to(device)
            domains = data["domain"].to(device)

            # if cfg.domain_method == "cdan":
            #     cls_pred_x, base_feat, domain_cls_pred = model(
            #         imgs, alpha, class_prob=torch.softmax(model(imgs, alpha)[0], dim=1)
            #     )
            # else:
            cls_pred_x, base_feat, domain_cls_pred = model(imgs, alpha)

            # === Classification loss/acc only for source domain ===
            if cfg.domain_method in ["dann", "cdan", "mmd"]:
                src_mask = (domains == 0)
                if src_mask.sum() > 0:
                    iloss = image_cls_loss(cls_pred_x[src_mask], cls_labels[src_mask])
                    img_loss += iloss.item() * src_mask.sum().item()
                    cls_correct += (cls_pred_x[src_mask].argmax(1) == cls_labels[src_mask]).sum().item()
                    total += src_mask.sum().item()
                else:
                    iloss = torch.tensor(0.0, device=imgs.device)
            else:
                img_loss += image_cls_loss(cls_pred_x, cls_labels)
                cls_correct += (cls_pred_x.argmax(1) == cls_labels).sum().item()
                total += cls_pred_x.size(0)

            # === Domain/MMD Loss ===
            if cfg.domain_method in ["dann", "cdan"]:
                dloss = domain_loss(domain_cls_pred, domains)
                dom_loss += dloss.item() * imgs.size(0)
                dom_correct += (domain_cls_pred.argmax(1) == domains).sum().item()
            elif cfg.domain_method == "mmd":
                unique_domains = domains.unique()
                mmd_batch = 0
                count = 0
                for i, dom_a in enumerate(unique_domains):
                    for dom_b in unique_domains[i+1:]:
                        idx_a = (domains == dom_a)
                        idx_b = (domains == dom_b)
                        if idx_a.sum() > 0 and idx_b.sum() > 0:
                            mmd_batch += mmd_loss(base_feat[idx_a], base_feat[idx_b])
                            count += 1
                if count > 0:
                    mmd_batch = mmd_batch / count
                else:
                    mmd_batch = torch.tensor(0.0, device=imgs.device)
                mmd_loss_total += mmd_batch.item() * imgs.size(0)

    # For DANN/CDAN: dom_loss/dom_acc over all samples; for MMD: mmd_loss_total
    dom_loss_value = (dom_loss / len(loader.dataset)) if cfg.domain_method in ["dann", "cdan"] \
        else (mmd_loss_total / len(loader.dataset) if cfg.domain_method == "mmd" else 0)
    dom_correct_value = (dom_correct / len(loader.dataset)) if cfg.domain_method in ["dann", "cdan"] else 0

    # Classification accuracy over source only
    img_loss_value = (img_loss / total) if total > 0 else 0
    cls_acc_value = (cls_correct / total) if total > 0 else 0

    return (
        img_loss_value,
        dom_loss_value,
        cls_acc_value,
        dom_correct_value,
    )



@timeit
def train():
    len_dataloader = len(train_loader)

    patience = getattr(cfg, "early_stop_patience", 10)
    min_delta = getattr(cfg, "early_stop_min_delta", 1e-4)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

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
            t_domain_labels = data["domain"].to(device)

            alpha = cfg.adv_alpha
            if cfg.schedule_alpha:
                p = float(i + epoch * len_dataloader) / (cfg.n_epochs * len_dataloader)
                alpha = 2. / (1. + np.exp(cfg.alpha_smoothing_factor * p)) - 1

            optimizer.zero_grad()
            if cfg.use_fp16:
                with autocast(device_type="cuda"):
                    # if cfg.domain_method == "cdan":
                    #     cls_pred_x, base_feat, domain_cls_pred = model(t_img, alpha)
                    # else:
                    cls_pred_x, base_feat, domain_cls_pred = model(t_img, alpha)

                    # === Classification loss/acc only for source domain ===
                    if cfg.domain_method in ["dann", "cdan", "mmd"]:
                        src_mask = (t_domain_labels == 0)
                        if src_mask.sum() > 0:
                            img_loss = image_cls_loss(cls_pred_x[src_mask], t_cls_labels[src_mask])
                        else:
                            img_loss = torch.tensor(0.0, device=t_img.device, requires_grad=True)
                        total_loss = img_loss
                    else:
                        img_loss = image_cls_loss(cls_pred_x, t_cls_labels)
                        total_loss = img_loss  # Non Adv Training

                    # === Domain/MMD Loss ===
                    if cfg.domain_method in ["dann", "cdan"]:
                        dom_loss = domain_loss(domain_cls_pred, t_domain_labels)
                        total_loss += dom_loss
                    elif cfg.domain_method == "mmd":
                        unique_domains = t_domain_labels.unique()
                        mmd_batch = 0
                        count = 0
                        for i, dom_a in enumerate(unique_domains):
                            for dom_b in unique_domains[i+1:]:
                                idx_a = (t_domain_labels == dom_a)
                                idx_b = (t_domain_labels == dom_b)
                                if idx_a.sum() > 0 and idx_b.sum() > 0:
                                    mmd_batch += mmd_loss(base_feat[idx_a], base_feat[idx_b])
                                    count += 1
                        if count > 0:
                            mmd_batch = mmd_batch / count
                        else:
                            mmd_batch = torch.tensor(0.0, device=t_img.device, requires_grad=True)
                        dom_loss = mmd_batch
                        total_loss += dom_loss
                    else:
                        dom_loss = torch.tensor(0.0, device=t_img.device, requires_grad=True)

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # if cfg.domain_method == "cdan":
                #     cls_pred_x, base_feat, domain_cls_pred = model(t_img, alpha)
                # else:
                cls_pred_x, base_feat, domain_cls_pred = model(t_img, alpha)

                # === Classification loss/acc only for source domain ===
                if cfg.domain_method in ["dann", "cdan", "mmd"]:
                    src_mask = (t_domain_labels == 0)
                    if src_mask.sum() > 0:
                        img_loss = image_cls_loss(cls_pred_x[src_mask], t_cls_labels[src_mask])
                    else:
                        img_loss = torch.tensor(0.0, device=t_img.device, requires_grad=True)
                    total_loss = img_loss
                else:
                    img_loss = image_cls_loss(cls_pred_x, t_cls_labels)
                    total_loss = img_loss # Non Adv Training

                # === Domain/MMD Loss ===
                if cfg.domain_method in ["dann", "cdan"]:
                    dom_loss = domain_loss(domain_cls_pred, t_domain_labels)
                    total_loss += dom_loss
                elif cfg.domain_method == "mmd":
                    unique_domains = t_domain_labels.unique()
                    mmd_batch = 0
                    count = 0
                    for i, dom_a in enumerate(unique_domains):
                        for dom_b in unique_domains[i+1:]:
                            idx_a = (t_domain_labels == dom_a)
                            idx_b = (t_domain_labels == dom_b)
                            if idx_a.sum() > 0 and idx_b.sum() > 0:
                                mmd_batch += mmd_loss(base_feat[idx_a], base_feat[idx_b])
                                count += 1
                    if count > 0:
                        mmd_batch = mmd_batch / count
                    else:
                        mmd_batch = torch.tensor(0.0, device=t_img.device, requires_grad=True)
                    dom_loss = mmd_batch
                    total_loss += dom_loss
                else:
                    dom_loss = torch.tensor(0.0, device=t_img.device, requires_grad=True)
                    

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # === Logging (source only for classification stats) ===
            if cfg.domain_method in ["dann", "cdan", "mmd"]:
                # Domain adaptation: classification only for source (domain==0)
                src_mask = (t_domain_labels == 0)
                if src_mask.sum() > 0:
                    img_loss = image_cls_loss(cls_pred_x[src_mask], t_cls_labels[src_mask])
                    train_img_loss += img_loss.item() * src_mask.sum().item()
                    train_cls_correct += (cls_pred_x[src_mask].argmax(1) == t_cls_labels[src_mask]).sum().item()
                    train_total += src_mask.sum().item()
                else:
                    img_loss = torch.tensor(0.0, device=t_img.device, requires_grad=True)
            else:
                # Non-adversarial: use all domains for classification
                img_loss = image_cls_loss(cls_pred_x, t_cls_labels)
                train_img_loss += img_loss.item() * t_img.size(0)
                train_cls_correct += (cls_pred_x.argmax(1) == t_cls_labels).sum().item()
                train_total += t_img.size(0)

            if cfg.domain_method in ["dann", "cdan"]:
                train_dom_correct += (domain_cls_pred.argmax(1) == t_domain_labels).sum().item()
            if cfg.domain_method in ["dann", "cdan", "mmd"]:
                train_dom_loss += dom_loss.item() * t_img.size(0)

        avg_train_img_loss = train_img_loss / train_total if train_total > 0 else 0
        avg_train_dom_loss = train_dom_loss / len(train_loader.dataset) if cfg.domain_method in ["dann", "cdan", "mmd"] else 0
        train_cls_acc = train_cls_correct / train_total if train_total > 0 else 0
        train_dom_acc = train_dom_correct / len(train_loader.dataset) if cfg.domain_method in ["dann", "cdan"] else 0

        log_dict = {
            "epoch": epoch + 1,
            "train/img_loss": avg_train_img_loss,
            "train/cls_acc": train_cls_acc,
        }
        if cfg.domain_method in ["dann", "cdan", "mmd"]:
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

        if epoch % 2 == 0 or epoch+1==cfg.n_epochs:
            val_results = eval_model(model, val_loader, device, image_cls_loss, domain_loss, alpha=0)
            test_results = eval_model(model, test_loader, device, image_cls_loss, domain_loss, alpha=0)

            log_dict.update({
                "val/img_loss": val_results[0],
                "val/cls_acc": val_results[2],
                "test/img_loss": test_results[0],
                "test/cls_acc": test_results[2],
            })
            if cfg.domain_method in ["dann", "cdan", "mmd"]:
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

            # --- Early Stopping Logic ---
            val_loss = val_results[0]
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # Save best weights
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    if best_weights is not None:
                        model.load_state_dict(best_weights)
                    break
                    
                # TODO SAVE BEST MODEL

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

    domain_loss = nn.CrossEntropyLoss().to(device)
    image_cls_loss = nn.CrossEntropyLoss().to(device)

    model = DomainAdaptModel(cfg).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler()
    train()