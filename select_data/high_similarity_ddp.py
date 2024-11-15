import os
import sys 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
project_dir = os.getcwd()
sys.path.append(project_dir)

import torch
import torch.nn.functional as F
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from torch.utils.data import DataLoader
#from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
from collections import defaultdict
from tqdm import tqdm

from src.utils import setup_ddp, cleanup_ddp, attachdebug, init_wandb, wandb_log, initialize_classification_head, set_seed
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def default_similarity():
    return {'similarity': -float('inf'), 'image_path': None, 'caption': None}

# 初始化分布式进程组
# def setup():
#     rank = int(os.environ['RANK'])  # 获取全局rank
#     local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank（进程绑定的 GPU）
#     world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
#     dist.init_process_group(backend='nccl', init_method='env://')
#     torch.cuda.set_device(rank)

#     # 打印当前进程所在的 GPU 信息
#     current_device = torch.cuda.current_device()
#     device_name = torch.cuda.get_device_name(current_device)
#     print(f"Process {rank}, Local Rank {local_rank} using GPU {current_device}: {device_name}")


# # 销毁分布式进程组
# def cleanup():
#     dist.destroy_process_group()

# 计算余弦相似度的批量操作
def compute_cosine_similarity_batch(image_features, text_features):
    return torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)

# 批次处理函数，批量计算相似度
def process_batch(ft_image_features, ft_text_features, labels, img_paths,  best_similarity_per_class):
    similarities = compute_cosine_similarity_batch(ft_image_features, ft_text_features)

    unique_labels = torch.unique(labels)
    for label in unique_labels:
        label_mask = (labels == label)
        label_similarities = similarities[label_mask]
        label_img_paths = [img_paths[i] for i in torch.nonzero(label_mask, as_tuple=True)[0]]

        max_idx = torch.argmax(label_similarities).item()
        if label_similarities[max_idx] > best_similarity_per_class[label.item()]['similarity']:
            best_similarity_per_class[label.item()] = {
                'similarity': label_similarities[max_idx].item(),
                'image_path': label_img_paths[max_idx],
            }

# 主函数，DDP 中每个进程运行的核心逻辑
# 使用hydra参数管理方式
@hydra.main(config_path=os.path.join(project_dir, "config/SelectData"), config_name="config.yaml")
def main_worker(cfg:DictConfig):
    #setup()
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
    ori_cwd = hydra.utils.get_original_cwd()
    cfg.ori_cwd = ori_cwd
    cfg.world_size = world_size
    #是否加当前工作目录，确定数据集的位置。
    cfg.data_location = os.path.join(cfg.ori_cwd, cfg.data_location) 
    # 初始化模型
    cket_path = os.path.join(cfg.ckpt_dir, "checkpoint_9.pth")#选取训练好的模型
    clip_encoder = CLIPEncoder(cfg, keep_lang=True).to(local_rank)
    clip_encoder.load(cket_path)
    #classification_head = ClassificationHead(normalize=True, weights=None).to(local_rank)

    #clip_encoder = torch.compile(clip_encoder)
    #classification_head = torch.compile(classification_head)

    # 使用 DDP 包装模型
    clip_encoder = DDP(clip_encoder, device_ids=[local_rank])
    #classification_head = DDP(classification_head, device_ids=[local_rank])

    clip_encoder.eval()
    #classification_head.eval()

    preprocess_fn = clip_encoder.module.val_preprocess

    datasets_class = getattr(datasets, cfg.dataset.name)#数据集类
    dataset = datasets_class(preprocess_fn,
                            location=cfg.data_location,
                            batch_size=cfg.batch_size,templetes=cfg.model.template)#获取数据集

    sampler = DistributedSampler(dataset.train_dataset, num_replicas=world_size, rank=rank, shuffle=False)#分布式数据获取

    loader = DataLoader(dataset.train_dataset, batch_size=cfg.eval_batch_size, sampler=sampler, num_workers=cfg.workers, pin_memory=False)
    # 保存每个类别的最高相似度样本对信息
    best_similarity_per_class = defaultdict(default_similarity)

    # 评估循环
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(loader), desc=f"Rank {rank} Processing Dataset", position=rank)):
            ft_image,ft_text,label,img_path =batch["images"].to(local_rank),batch["captions"].to(local_rank),batch["labels"].to(local_rank),batch["image_paths"]
            ft_image_features, ft_text_features, _ = clip_encoder(ft_image, ft_text)
            process_batch(ft_image_features, ft_text_features, label, img_path, best_similarity_per_class)

    # with torch.no_grad():
    #     for ft_image, ft_text, label, img_path, caption in tqdm(ft_dataloader, total=len(ft_dataloader), desc=f"Rank {rank} Processing Dataset", position=rank):
    #         ft_image, ft_text, label = ft_image.cuda(local_rank), ft_text.cuda(local_rank), label.cuda(local_rank)
    #         ft_image_features, ft_text_features, _ = clip_encoder(ft_image, ft_text)
    #         process_batch(ft_image_features, ft_text_features, label, img_path, caption, best_similarity_per_class)

    # 同步所有 GPU 的结果
    dist.barrier()
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, best_similarity_per_class)

    # 仅在主进程（rank 0）中保存 CSV 文件
    if rank == 0:
        combined_data = defaultdict(lambda: {'similarity': -float('inf'), 'image_path': None})

        for gpu_data in gathered_data:
            for label, info in gpu_data.items():
                if info['similarity'] > combined_data[label]['similarity']:
                    combined_data[label] = info

        # 保存为 CSV
        data = []
        for label, info in combined_data.items():
            data.append({
                'label': label,
                'similarity': info['similarity'],
                'image_path': info['image_path']
            })

        df = pd.DataFrame(data)
        df.to_csv('datasets/csv/select_data/prob/best_similarity_per_class.csv', index=False)


if __name__ == '__main__':
    setup_ddp()
    #cfg = parse_arguments()  # 解析输入参数
    # world_size = torch.cuda.device_count()  # 获取GPU数量
    main_worker()  # 主进程运行
    cleanup_ddp()
    