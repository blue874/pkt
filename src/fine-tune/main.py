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


@hydra.main(config_path=os.path.join(project_dir, "config/FT"), config_name="config.yaml")
def main_worker(cfg: DictConfig):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)
    ori_cwd = hydra.utils.get_original_cwd()
    cfg.ori_cwd = ori_cwd
    cfg.dataset.csv_file = os.path.join(cfg.ori_cwd, cfg.dataset.csv_file)
    #cfg.data_location = os.path.join(cfg.ori_cwd, cfg.data_location)(不加当前工作目录)
    print("Configuration")
    print(OmegaConf.to_yaml(cfg))
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
    cfg.world_size = world_size
    cfg.batch_size_per_gpu = cfg.batch_size // world_size  # 每个 GPU 的 batch size

    clip_encoder = CLIPEncoder(cfg, keep_lang=True).to(local_rank)
    clip_encoder = DDP(clip_encoder, device_ids=[local_rank])
    init_wandb(cfg, model_list=[clip_encoder.module])

    train_preprocess = clip_encoder.module.train_preprocess
    val_preprocess = clip_encoder.module.val_preprocess


    if cfg.loss.name == "ClipLoss":
        loss_fn = ClipLoss(local_loss=cfg.loss.local_loss,
                                gather_with_grad=cfg.loss.gather_with_grad,
                                cache_labels=cfg.loss.cache_labels,
                                rank=local_rank,
                                world_size=world_size,
                                use_horovod=cfg.loss.use_horovod)
    else:
        assert False, "Loss not implemented"


    train_dataset_class = getattr(datasets, cfg.dataset.name)
    train_dataset = train_dataset_class(train_preprocess,
                            location=cfg.data_location,
                            batch_size=cfg.batch_size_per_gpu,templetes=cfg.model.template)
    train_dataset = train_dataset.train_dataset

    # train_dataset = CsvDataset(cfg.dataset.csv_file, train_preprocess, img_key=cfg.dataset.csv_img_key, caption_key=cfg.dataset.csv_caption_key, sep=cfg.dataset.csv_separator, label_key=cfg.dataset.supervised_label_key)
    
    sample = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size_per_gpu, sampler=sample, num_workers=cfg.workers, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id))

    total_params = list(clip_encoder.module.parameters())
    params = [p for p in total_params if p.requires_grad]
    

    optimizer = torch.optim.AdamW(params, lr=cfg.optimizer.learn_rate, weight_decay=cfg.optimizer.weight_decay)
    scheduler = cosine_lr(optimizer, cfg.scheduler.base_lrs, cfg.scheduler.warmup_length, cfg.epochs * (len(train_dataloader)), cfg.scheduler.min_lr)

    stats = []
    for epoch in range(-1, cfg.epochs):
        print(f"Epoch {epoch}")
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        id_flyp_loss_sum = 0
        clip_encoder.train()
        clip_encoder = clip_encoder.cuda()

        if epoch != -1:
            for i, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc=f"Rank {rank} Processing Dataset", position=rank)):
                step = i + epoch * len(train_dataloader)
                optimizer.zero_grad()
                scheduler(step)
                #print(batch["captions"])
                #exit(0)
                ft_image, ft_text = batch["images"].cuda(), batch["captions"].cuda()

                ft_image_features, ft_text_features, logit_scale2 = clip_encoder(ft_image, ft_text)
                
                loss = loss_fn(ft_image_features, ft_text_features, logit_scale2)

                loss.backward()
                optimizer.step()

                id_flyp_loss_sum += loss.item()

                wandb_log({
                    "train/Contrastive Loss": loss.item(),
                    "train/Epoch": epoch + 1,
                    "train/Step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                    })
                
                scheduler(step)

        # Evaluate
        test_result = {f"test/{dataset_name}": 0 for dataset_name in cfg.eval_datasets}
        for i, dataset_name in enumerate(cfg.eval_datasets):
            print('Evaluating on', dataset_name)
            dataset_class = getattr(datasets, dataset_name)
            test_dataset = dataset_class(clip_encoder.module.val_preprocess,
                                    location=cfg.data_location,
                                    batch_size=cfg.batch_size)
            test_dataset = test_dataset.test_dataset
            sample = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

            test_dataloader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, sampler=sample, num_workers=cfg.workers, pin_memory=True)


            clip_encoder.eval()
            classification_head = initialize_classification_head(cfg, clip_encoder.module, dataset_name).cuda()
            classification_head = torch.compile(classification_head)
            classification_head = DDP(classification_head, device_ids=[local_rank])
            classification_head.eval()

            results = {"correct": 0, "n": 0}
            with torch.no_grad():
                for i, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc=f"Rank {rank} Processing Dataset", position=rank)):
                    image, label = batch["images"].to(local_rank), batch["labels"].to(local_rank)
                    logits = get_logits(image, clip_encoder, classification_head)
                    # count the number of correct predictions
                    correct = (logits.argmax(dim=-1) == label).sum().item()
                    results["correct"] += correct
                    results["n"] += len(label)

            dist.barrier()
            gathered_data = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_data, results)

            if rank == 0:
                combined_data = {"correct": 0, "n": 0}
                for gpu_data in gathered_data:
                    combined_data["correct"] += gpu_data["correct"]
                    combined_data["n"] += gpu_data["n"]
                top1 = combined_data['correct']/combined_data['n']
                print(f"Top1: {top1}")
                test_result[f"test/{dataset_name}"] = top1

                epoch_stats[f"{dataset_name}"] = round(top1*100, 2)

        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)

        stats_df.to_csv(os.path.join(cfg.log_dir, 'stats.tsv'), sep='\t')

        wandb_log(test_result | {"test/Epoch": epoch + 1})

        if cfg.ckpt_dir is not None and epoch != -1 and local_rank == 0:
            model_path = os.path.join(cfg.ckpt_dir, f'checkpoint_{epoch}.pth')
            clip_encoder.module.save(model_path)
            optim_path = os.path.join(cfg.ckpt_dir, f'optimizer_{epoch}.pth')
            torch.save(optimizer.state_dict(), optim_path)

        dist.barrier()

        print(f"Epoch {epoch} finished")
        print(test_result)

if __name__ == "__main__":
    setup_ddp()
    # attachdebug()
    main_worker()
    cleanup_ddp()