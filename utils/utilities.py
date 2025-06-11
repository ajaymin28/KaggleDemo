import os
from os import listdir
import random
import glob
import torch
import torch.distributed as dist
import sys
from functools import wraps
import time

DEBUG = False


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper

def getrandomIndexesForImages(gallery_image_count=10, query_image_count=5, max_idx_of_images=150):
    """
    DO NOT CHANGE THE SEEDS
    """
    # Fixed seed for repetitive results (DO NOT CHANGE)
    const_seed_q = 200
    const_seed_g = 100
    print(f"Q seed: {const_seed_q} G seed: {const_seed_g}")
    # Final number of values 
    g_number = gallery_image_count
    q_number = query_image_count
    # Seed and retrieve the values
    random.seed(const_seed_q)
    query_indexes = [random.randint(0, max_idx_of_images) for i in range(0, q_number)]

    # Seed and retrieve the values
    random.seed(const_seed_g)
    gallery_indexes = [random.randint(0, max_idx_of_images) for i in range(0, g_number)]

    # print(query_indexes)
    # print(gallery_indexes)
    return gallery_indexes, query_indexes

def getClassesNames(gallery_images_path,query_images_path):
    return list(set(listdir(gallery_images_path) + listdir(query_images_path)))

def getData(images_path, classes_names):
    classes_idnames = {}
    classes_nameids = {}
    images_paths = []
    image_labels = []
    for classid, cls_name in enumerate(classes_names):
        class_path = os.path.join(images_path, cls_name, "*.jpg")
        class_images_paths = glob.glob(class_path)
        for ip in class_images_paths:
            images_paths.append(ip)
            image_labels.append(cls_name)

        if classid not in classes_idnames.keys():
            classes_idnames[classid] = cls_name
            classes_nameids[cls_name] = classid
    return images_paths, image_labels, classes_idnames, classes_nameids


def getImagePaths(gallery_images_path, query_images_path, gallery_image_count=10, query_image_count=5, max_idx_of_images=150):
    gallery_idxs, query_idxs = getrandomIndexesForImages(gallery_image_count=gallery_image_count, query_image_count=query_image_count,max_idx_of_images=max_idx_of_images)
    classes_names = listdir(gallery_images_path)
    print("classes",classes_names)

    images_paths = {}
    for cls_name in classes_names:
        if cls_name not in images_paths.keys():
            images_paths[cls_name] = {"gallery": [], "query": []}
        class_path = os.path.join(gallery_images_path, cls_name, "*.jpg")
        class_images_paths = glob.glob(class_path)
        for gal_idx in gallery_idxs:
            images_paths[cls_name]["gallery"].append(class_images_paths[gal_idx])

        class_path_q = os.path.join(query_images_path, cls_name, "*.jpg")
        class_images_paths_q = glob.glob(class_path_q)
        for que_idx in query_idxs:
            images_paths[cls_name]["query"].append(class_images_paths_q[que_idx])
            
    return images_paths


"""
Copied from 

https://github.com/ajaymin28/DinIE/blob/main/utils2/utils.py

"""

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = get_world_size()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        # backend="gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


import argparse

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_args_parser():
    parser = argparse.ArgumentParser('PytorchHandsOn', add_help=False)


    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.01, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--optimizer', default='sgd', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Misc
    parser.add_argument('--data_path', default='./dataset', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./output", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument('--port', default=29507, type=int, help='Port used for Dist training')
    return parser

    
"""
Copied from ENDs

https://github.com/ajaymin28/DinIE/blob/main/utils2/utils.py

"""

def is_wandb_logged_in():
    import os
    import wandb
    # Check environment
    if not os.environ.get("WANDB_API_KEY"):
        try:
            # Try the API call
            _ = wandb.Api()
            return True
        except Exception:
            return False
    return True

def config_to_dict(obj):
    # Get all relevant attributes (class + instance, skip dunder and callables)
    keys = [
        k for k in set(obj.__class__.__dict__.keys()).union(vars(obj).keys())
        if not k.startswith("__") and not callable(getattr(obj, k, None))
    ]
    def to_serializable(v):
        if isinstance(v, torch.device):
            return str(v)
        return v
    return {k: to_serializable(getattr(obj, k)) for k in keys}