import torch
import random
import numpy as np
from tqdm import tqdm
import torchvision
import torch.nn as nn

from utils.models import MyNet

import wandb
import os

from utils.utilities import get_args_parser, init_distributed_mode, get_world_size,get_rank
import argparse
from pathlib import Path
import torch.distributed as dist


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def step_val(model, dataset, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for idx, (data, labels) in enumerate(dataset):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(data)
        loss = loss_fn(preds, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        print(f"Val processed {idx}/{len(dataset)}")

    # Aggregate metrics across all processes (DDP)
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)

    # Sum across all ranks
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    avg_acc = total_correct_tensor.item() / total_samples_tensor.item()

    return avg_acc, avg_loss

import time

def step_train_profiled(model, dataset, loss_fn, optimizer, device, profile_batches=10):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    time_stats = {
        "data_to_device": 0,
        "forward": 0,
        "loss": 0,
        "optimizer_zero_grad": 0,
        "backward": 0,
        "optimizer_step": 0,
        "cuda_synchronize": 0,
        "all_reduce": 0
    }

    batch_count = 0
    for batch_idx, (data, labels) in enumerate(dataset):
        if batch_idx >= profile_batches:
            break
        tic = time.time()
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        time_stats["data_to_device"] += time.time() - tic

        tic = time.time()
        preds = model(data)
        time_stats["forward"] += time.time() - tic

        tic = time.time()
        loss = loss_fn(preds, labels)
        time_stats["loss"] += time.time() - tic

        tic = time.time()
        optimizer.zero_grad()
        time_stats["optimizer_zero_grad"] += time.time() - tic

        tic = time.time()
        loss.backward()
        time_stats["backward"] += time.time() - tic

        tic = time.time()
        optimizer.step()
        time_stats["optimizer_step"] += time.time() - tic

        tic = time.time()
        torch.cuda.synchronize()
        time_stats["cuda_synchronize"] += time.time() - tic

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        batch_count += 1


    # DDP metric reduce
    tic = time.time()
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    time_stats["all_reduce"] += time.time() - tic

    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    avg_acc = total_correct_tensor.item() / total_samples_tensor.item()

    # Print timing results
    print("\nProfiling results (average per batch):")
    for k, v in time_stats.items():
        # For all_reduce, show total (since it's one call)
        if k == "all_reduce":
            print(f"{k:22s}: {v:.6f} s (total for reduce)")
        else:
            print(f"{k:22s}: {v/batch_count:.6f} s")
    print("--------------------------------------------------")

    return avg_acc, avg_loss, optimizer



def step_train(model, dataset, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for idx, (data, labels) in enumerate(dataset):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(data)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    # DDP: Aggregate metrics
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    avg_acc = total_correct_tensor.item() / total_samples_tensor.item()

    return avg_acc, avg_loss, optimizer


def step(model, dataset, loss_fn, optimizer=None, isVal=False):
    t_loss = 0
    acc, t = 0, 1
    if isVal:
        model.eval()
    else:
        model.train()

    for (data, labels) in dataset:
        data = data.to(device)
        labels = labels.to(device)
        preds = model(data)
        loss = loss_fn(preds, labels)

        if not isVal:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        t_loss += loss.item()
        preds_soft = nn.functional.softmax(preds, dim=-1)
        idx = preds_soft.argmax(dim=-1)
        acc += (labels==idx).sum().item()
        t += labels.size(0)

    torch.cuda.synchronize()
    
    t_loss /= len(dataset)
    acc = acc/t

    if isVal:
        return acc,t_loss, None
    else:
        return acc,t_loss, optimizer

if __name__=="__main__":


    parser = argparse.ArgumentParser('PytorchHandsOn', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29507'
    # os.environ["GLOO_USE_LIBUV"] = "0"
    # os.environ["USE_LIBUV"] = "0"

    init_distributed_mode(args)

    EPOCHS = args.epochs
    LR = args.lr
    BATCH = args.batch_size_per_gpu

    world_size = get_world_size()
    num_workers = min(8, os.cpu_count() // world_size)
    # num_workers = 0

    # print(f"DDP init: rank {rank}, local_rank {local_rank}, world_size {world_size} workers: {num_workers}")

    set_seed(args.seed)

    transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.transforms.RandomCrop(size=(32, 32)),
            torchvision.transforms.transforms.GaussianBlur(kernel_size=(3,3)),
            torchvision.transforms.transforms.ColorJitter(),
            torchvision.transforms.transforms.RandomAutocontrast(),
            torchvision.transforms.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    dataset = torchvision.datasets.CIFAR10(root="./datasets", transform=transform, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(root="./datasets", transform=transform, train=False, download=True)

    total_samples = len(dataset)
    val_samples = int(0.1 * total_samples)
    train_samples = total_samples - val_samples
    # print(train_samples, val_samples)
    train_ds, val_ds = torch.utils.data.random_split(dataset,[train_samples,val_samples])
    str_classes = dataset.classes
    # # t_str_classes = test_dataset.classes


    sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True)
    train_ds = torch.utils.data.DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=BATCH,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


    val_sampler = torch.utils.data.DistributedSampler(val_ds, shuffle=False)
    val_ds = torch.utils.data.DataLoader(
        val_ds,
        sampler=val_sampler,
        batch_size=BATCH,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


    test_sampler = torch.utils.data.DistributedSampler(test_ds, shuffle=False)
    test_ds = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=BATCH,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = MyNet(n_classes=10).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.01, momentum=0.9,nesterov=True)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # run = wandb.init(
    #     project="cifar_test",  
    #     config={ 
    #         "learning_rate": LR,
    #         "epochs": EPOCHS,
    #     },
    # )
    
    for epoch in tqdm(range(EPOCHS)):
        train_ds.sampler.set_epoch(epoch)
        val_ds.sampler.set_epoch(epoch)

        tt_loss, tv_loss = 0, 0

        # acc, tt_loss, optimizer = step(model, train_ds, loss_fn, optimizer=optimizer, isVal=False)
        # vacc, tv_loss, _ = step(model, val_ds, loss_fn, isVal=True)

        train_acc, train_loss, optimizer = step_train_profiled(model, train_ds, loss_fn, optimizer, device)
        val_acc, val_loss = step_val(model, val_ds, loss_fn, device)

        if dist.get_rank() == 0:
            print(f"Epoch[{epoch}/{EPOCHS}] train_loss: {train_loss:.4f} acc:{train_acc:.2f} val_loss: {val_loss:.4f} val_acc:{val_acc:.2f}")

        # wandb.log({"accuracy": acc, "loss": tt_loss, "v_loss": tv_loss, "v_acc": vacc})
        # print(f"Epoch[{epoch}/{EPOCHS}] loss: {tt_loss:.4f} acc:{acc:.2f} vloss: {tv_loss:.4f} vacc:{vacc:.2f}")

    dist.destroy_process_group()