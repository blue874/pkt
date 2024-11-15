import debugpy
import os
import random
import numpy as np
import torch
import torch.distributed as dist

import wandb
from omegaconf import DictConfig, OmegaConf

from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead

def attachdebug(rank=0):
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0  # 对于单卡环境，默认为 0
    
    # 设置调试服务器的端口
    if local_rank == rank:
        os.environ["DEBUG_MODE"] = "1"
        debugpy.listen(("0.0.0.0", 34252 + local_rank))  # 每个进程用不同的端口
        print(f"Waiting for debugger to attach on rank {local_rank}...")
        debugpy.wait_for_client()  # 让进程暂停，等待调试器连接


# 初始化分布式进程组
def setup_ddp():
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank（进程绑定的 GPU）
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)

    # 打印当前进程所在的 GPU 信息
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Process {rank}, Local Rank {local_rank} using GPU {current_device}: {device_name}")


def init_wandb(cfg, model_list):
    if dist.get_rank() == 0:
        if os.getenv("DEBUG_MODE") == "1":
            print("Not initializing wandb in debug mode")
            wandb.init(mode="disabled")
        else:
            wandb.init(
                project=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),#转字典
                dir=cfg.logging.dir,
                mode=cfg.logging.mode,
                name=cfg.logging.name,
                save_code=True
            )
        # for model in model_list:
        #     wandb.watch(model, log="all")
    else:
        wandb.init(mode="disabled")

def wandb_log(log_dict):
    if dist.get_rank() == 0:
        wandb.log(log_dict)

# 广播分类头的参数
def broadcast_classification_head_parameters(classification_head):
    # 遍历所有参数，并从 rank 0 广播到其他进程
    for param in classification_head.parameters():
        dist.broadcast(param.data, src=0)  # 将参数的值从 rank 0 广播到其他进程

# 初始化分类头
def initialize_classification_head(args, clip_encoder, dataset=None):
    classification_head = get_zeroshot_classifier(args, clip_encoder.model, dataset) 
    # if rank == 0:
    #     # 仅在 Rank 0 上初始化分类头
    #     classification_head = get_zeroshot_classifier(args, clip_encoder.model, dataset)
    # else:
    #     if dataset=="ImageNetR" or dataset=="ImageNetA":
    #         shape = [512, 200]
    #     elif dataset == "ImageNet" or dataset == "ImageNetV2" or dataset == "ImageNetSketch":
    #         shape = [512, 1000]
    #     else:
    #         AssertionError("Dataset not implemented")
    #     classification_head = ClassificationHead(normalize=True, weights=None, shape=shape)  # 其他进程暂时为 None

    # classification_head = classification_head.to(rank)  # 将分类头移动到当前进程的 GPU

    # broadcast_classification_head_parameters(classification_head)
    
    return classification_head

# 销毁分布式进程组
def cleanup_ddp():
    dist.destroy_process_group()

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        