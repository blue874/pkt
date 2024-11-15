import os
import sys 
project_dir = os.getcwd()
sys.path.append(project_dir)
import init_print

import numpy as np
import pandas as pd
import torch
from clip.loss import ClipLoss
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.models.eval import evaluate
from src.models.flyp_loss import flyp_loss
from src.models.ce_ablation import ce_ablation
from src.models.utils import fisher_load
from src.args import parse_arguments
import logging

from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data, CsvDataset
import src.datasets as datasets
from collections import defaultdict

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


from src.utils import setup_ddp, cleanup_ddp, attachdebug, init_wandb, wandb_log, initialize_classification_head, set_seed

from src.models.utils import get_logits
#这是选取每个类logits最大的样本作为训练集
def default_similarity():
    return {'probability': -float('inf'), 'image_path': None}#返回默认字典



@hydra.main(config_path=os.path.join(project_dir, "config/SelectData"), config_name="config.yaml")
def main_worker(cfg: DictConfig):
    set_seed(cfg.seed)
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
    cfg.world_size = world_size

    clip_encoder = CLIPEncoder(cfg, keep_lang=True).to(local_rank)#放到当前GPU上
    cket_path = os.path.join(cfg.ckpt_dir, "checkpoint_9.pth")#选取训练好的模型
    clip_encoder.load(cket_path)#导入
    clip_encoder.eval()#model调成评估模式

    classification_head = get_zeroshot_classifier(cfg, clip_encoder.model, dataset=cfg.dataset.name)#定义分类头
    classification_head.eval()#评估模式

    # clip_encoder = torch.compile(clip_encoder)
    # classification_head = torch.compile(classification_head)
    
    preprocess_fn = clip_encoder.val_preprocess#获取预处理方式

    datasets_class = getattr(datasets, cfg.dataset.name)#数据集类
    dataset = datasets_class(preprocess_fn,
                            location=cfg.data_location,
                            batch_size=cfg.batch_size,templetes=cfg.model.template)#获取数据集

    sampler = DistributedSampler(dataset.train_dataset, num_replicas=world_size, rank=rank, shuffle=False)#分布式数据获取

    loader = DataLoader(dataset.train_dataset, batch_size=cfg.eval_batch_size, sampler=sampler, num_workers=cfg.workers, pin_memory=False)

    best_similarity_per_class = defaultdict(default_similarity)#初始化选取数据列表

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(loader), desc=f"Rank {rank} Processing Dataset", position=rank)):
            images, label, img_path,title = batch["images"].to(local_rank), batch["labels"].to(local_rank), batch["image_paths"],batch["captions"]
            logits = get_logits(images, clip_encoder, classification_head)#未归一化置信度
            probs = logits.softmax(dim=-1)#softmax归一化
            process_batch(probs, label, img_path, title, best_similarity_per_class)#计算相似度

    
    dist.barrier()#等待进程都结束
    gathered_data = [None for _ in range(world_size)]#初始化汇总列表
    dist.all_gather_object(gathered_data, best_similarity_per_class)

    if rank == 0:
        combined_data = defaultdict(lambda: {'probability': -float('inf'), 'image_path': None})

        for gpu_data in gathered_data:
            for label, info in gpu_data.items():
                if info['probability'] > combined_data[label]['probability']:
                    combined_data[label] = info
        data = []
        for label, info in combined_data.items():
            data.append({
                'title':info['title'],
                'image_path': info['image_path'],
                'label': label,
                'probability': info['probability']
            })
        
        #TODO: check to save as csv or not
        

        df = pd.DataFrame(data)
        df.to_csv('datasets/csv/select_data/caption/best_probability_per_class_VIT_B32.csv', index=False)


# 批次处理函数，批量计算相似度
def process_batch(probabilities, labels, img_paths, title,best_similarity_per_class):

    # probabilities B x C, labels B
    # get the probabilities of the correct labels
    probabilities = probabilities[torch.arange(probabilities.size(0)), labels]#获取每个类的可能性

    unique_labels = torch.unique(labels)#1去除重复标签
    for label in unique_labels:
        label_mask = (labels == label)#标识label位置独热向量（0,1，0，0）
        label_similarities = probabilities[label_mask]#找到对应类所有样本的可能性
        label_img_paths = [img_paths[i] for i in torch.nonzero(label_mask, as_tuple=True)[0]]#获取图像路径\

        label_title=[title[i] for i in torch.nonzero(label_mask,as_tuple=True)[0]]#

        max_idx = torch.argmax(label_similarities).item()#找最大值索引
        if label_similarities[max_idx] > best_similarity_per_class[label.item()]['probability']:
            best_similarity_per_class[label.item()] = {
                'probability': label_similarities[max_idx].item(),
                'image_path': label_img_paths[max_idx],
                'title':label_title[max_idx]#
            }

if __name__ == '__main__':
    setup_ddp()  # 初始化 DDP
    # attachdebug()  # 调试模式
    # args = parse_arguments()  # 解析输入参数
    # world_size = torch.cuda.device_count()  # 获取GPU数量
    main_worker()  # 主进程运行
    cleanup_ddp()  # 清理 DDP
    