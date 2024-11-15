#### distilling fine-tuning #########
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import sys 
project_dir = os.getcwd()
sys.path.append(project_dir)
import init_print
import torch
from clip.loss import ClipLoss
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data, CsvDataset
import src.datasets as datasets
from collections import defaultdict
from tqdm import tqdm
from losses import DistLogits
from rktloss import RkdDistance,RKdAngle,Hadamardloss,LogitsDiv,KLloss

from src.utils import setup_ddp, cleanup_ddp, attachdebug, init_wandb, wandb_log, initialize_classification_head, set_seed


@hydra.main(config_path=os.path.join(project_dir, "config/DIST"), config_name="config.yaml")
def main_worker(cfg: DictConfig):

    
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)
    ori_cwd = hydra.utils.get_original_cwd()#获取工作目录
    cfg.ori_cwd = ori_cwd
    #测试当前工作目录
    print(ori_cwd)
    cfg.dataset.csv_file = os.path.join(cfg.ori_cwd, cfg.dataset.csv_file)#选取用哪个数据集
    cfg.data_location = os.path.join(cfg.ori_cwd, cfg.data_location)#获取数据集位置
    print("Configuration")
    print(OmegaConf.to_yaml(cfg))
    rank = int(os.environ['RANK'])  # 获取全局rank
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取 local_rank
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数
    cfg.world_size = world_size
    cfg.batch_size_per_gpu = cfg.batch_size // world_size  # 每个 GPU 的 batch size

    #定义教师，学生模型
    clip_encoder_t = CLIPEncoder(cfg, keep_lang=True).to(local_rank)#教师模型初始化
    cket_path_t = os.path.join(cfg.ckpt_dir_t, "checkpoint_9.pth")#教师模型位置
    clip_encoder_t.load(cket_path_t)
    clip_encoder_t = DDP(clip_encoder_t, device_ids=[local_rank])#包装为分布式
    #不计算梯度
    for param in clip_encoder_t.module.parameters():
        param.requires_grad = False

    if cfg.loss.dist_type=="div_soft":
        clip_encoder_t_z=CLIPEncoder(cfg, keep_lang=True).to(local_rank)#零样本32
        clip_encoder_t_z= DDP(clip_encoder_t_z, device_ids=[local_rank])#包装为分布式
        for param in clip_encoder_t_z.module.parameters():
            param.requires_grad = False

    #学生模型
    cfg.model.backbone="ViT-B/16"
    clip_encoder_s = CLIPEncoder(cfg, keep_lang=True).to(local_rank)#学生模型初始化
    clip_encoder_s = DDP(clip_encoder_s, device_ids=[local_rank])#包装为分布式

    if cfg.loss.dist_type=="div_soft":
        clip_encoder_s_z=CLIPEncoder(cfg, keep_lang=True).to(local_rank)#零样本16
        clip_encoder_s_z= DDP(clip_encoder_s_z, device_ids=[local_rank])#包装为分布式
        for param in clip_encoder_s_z.module.parameters():
            param.requires_grad = False
    
    #日志
    init_wandb(cfg, model_list=[clip_encoder_s.module])
    #预处理方式
    train_preprocess = clip_encoder_s.module.train_preprocess
    val_preprocess = clip_encoder_s.module.val_preprocess
    #logit定义
    logit_fn = DistLogits(local_loss=cfg.loss.local_loss,
                                gather_with_grad=cfg.loss.gather_with_grad,
                                cache_labels=cfg.loss.cache_labels,
                                rank=local_rank,
                                world_size=world_size,
                                use_horovod=cfg.loss.use_horovod)
    if cfg.loss.loss_type=="flyp":
        #flyp损失
        clip_loss_fn = ClipLoss(local_loss=cfg.loss.local_loss,
                                gather_with_grad=cfg.loss.gather_with_grad,
                                cache_labels=cfg.loss.cache_labels,
                                rank=local_rank,
                                world_size=world_size,
                                use_horovod=cfg.loss.use_horovod)
    
    
    #数据集
    train_dataset = CsvDataset(cfg.dataset.csv_file, train_preprocess, img_key=cfg.dataset.csv_img_key, caption_key=cfg.dataset.csv_caption_key, sep=cfg.dataset.csv_separator, label_key=cfg.dataset.supervised_label_key)
    #分布式分数据
    sample = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    #加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size_per_gpu, sampler=sample, num_workers=cfg.workers, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id))

    #要优化参数列表,学生模型
    total_params = list(clip_encoder_s.module.parameters())
    #只更新需要梯度的参数
    params = [p for p in total_params if p.requires_grad]


    #优化器定义
    optimizer = torch.optim.AdamW(params, lr=cfg.optimizer.learn_rate, weight_decay=cfg.optimizer.weight_decay)
    #余弦学习率缩减
    scheduler = cosine_lr(optimizer, cfg.scheduler.base_lrs, cfg.scheduler.warmup_length, cfg.epochs * (len(train_dataloader)), cfg.scheduler.min_lr)

    stats = []
    for epoch in range(-1, cfg.epochs):
        print(f"Epoch {epoch}")
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        #loss_sum = 0
        clip_encoder_s.train()
        clip_encoder_t.eval()
        if cfg.loss.dist_type=="div_soft":
            clip_encoder_s_z.eval()
            clip_encoder_t_z.eval()
        

        if epoch != -1:
            for i, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc=f"Rank {rank} Processing Dataset", position=rank)):
                step = i + epoch * len(train_dataloader)

                optimizer.zero_grad()

                image, text, _, _, _ = batch

                image, text = image.to(local_rank), text.to(local_rank)#放到GPU
                image_feature_s, text_feature_s, logit_scale2_s = clip_encoder_s(image, text)#获取嵌入向量student
                image_feature_t, text_feature_t, logit_scale2_t = clip_encoder_t(image, text)#teaher

                t=cfg.loss.dist_tau#温度

                if cfg.loss.dist_type=="soft":
                    logit_s_image,logit_s_text=logit_fn(image_feature_s, text_feature_s, logit_scale2_s)
                    logit_t_image,logit_t_text=logit_fn(image_feature_t, text_feature_t, logit_scale2_t)

                    distloss_t=KLloss(logit_s_text,logit_t_text,t)
                    distloss_i=KLloss(logit_s_image,logit_t_image,t)
                    dist_loss=(distloss_t+distloss_i)/2

                    clip_loss = clip_loss_fn(image_feature_s, text_feature_s, logit_scale2_s)
                    loss=dist_loss*cfg.loss.dist_alpha+(1-cfg.loss.dist_alpha)*clip_loss
                    loss.backward()
                elif cfg.loss.dist_type=="div_soft":
                    image_feature_s_z, text_feature_s_z, logit_scale2_s_z = clip_encoder_s_z(image, text)#获取嵌入向量student,0样本
                    image_feature_t_z, text_feature_t_z, logit_scale2_t_z = clip_encoder_t_z(image, text)#teaher,0样本

                    logit_s_image,logit_s_text=logit_fn(image_feature_s, text_feature_s, logit_scale2_s)#学生logits
                    logit_t_image,logit_t_text=logit_fn(image_feature_t, text_feature_t, logit_scale2_t)#教师logits

                    logit_s_image_z,logit_s_text_z=logit_fn(image_feature_s_z, text_feature_s_z, logit_scale2_s_z)#零样本student logits
                    logit_t_image_z,logit_t_text_z=logit_fn(image_feature_t_z, text_feature_t_z, logit_scale2_t_z)#零样本teacher logits

                    pred_logit_image=LogitsDiv(logit_t_image,logit_t_image_z,logit_s_image_z)#预测image置信度
                    pred_logit_text=LogitsDiv(logit_t_text,logit_t_text_z,logit_s_text_z)#预测text置信度

                    distloss_i=KLloss(logit_s_image,pred_logit_image,t)
                    distloss_t=KLloss(logit_s_text,pred_logit_text,t)
                    dist_loss=(distloss_t+distloss_i)/2

                    clip_loss = clip_loss_fn(image_feature_s, text_feature_s, logit_scale2_s)
                    loss=dist_loss*cfg.loss.dist_alpha+(1-cfg.loss.dist_alpha)*clip_loss
                    loss.backward()

                else:
                    assert False, "Loss not implemented"
                
                #loss_sum +=loss.item()#计算总损失
                optimizer.step()#执行优化

                wandb_log({
                    "train/Contrastive Loss": clip_loss.item(),
                    "train/Distilling Loss": dist_loss.item(),
                    "train/Epoch": epoch + 1,
                    "train/Step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                    })
                
                scheduler(step)

        # test per epoch，评估
        test_dataset_name = None
        test_result = {f"test/{dataset_name}": 0 for dataset_name in cfg.eval_datasets}#初始化测试结果字典
        for i, dataset_name in enumerate(cfg.eval_datasets):
            # if 'Test' in dataset_name:
            test_dataset_name = dataset_name

            assert test_dataset_name is not None, 'please give test data'
            print('Evaluating on', test_dataset_name)
            test_dataset_class = getattr(datasets, test_dataset_name)
            test_dataset = test_dataset_class(clip_encoder_s.module.val_preprocess,
                                            location=cfg.data_location,
                                            batch_size=cfg.batch_size)

            test_dataset = test_dataset.test_dataset
            #分布式的数据分配
            sample = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            #分布式数据加载
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, sampler=sample, num_workers=cfg.workers, pin_memory=True)


            clip_encoder_s.eval()
            #分类头模型
            classification_head = initialize_classification_head(cfg, clip_encoder_s.module, test_dataset_name).cuda()
            classification_head = torch.compile(classification_head)
            classification_head = DDP(classification_head, device_ids=[local_rank])

            results = {"correct": 0, "n": 0}
            with torch.no_grad():
                for i, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc=f"Rank {rank} Processing Dataset", position=rank)):
                    image, label = batch["images"].to(local_rank), batch["labels"].to(local_rank)#放到GPU上
                    logits = get_logits(image, clip_encoder_s, classification_head)#计算置信度
                    # count the number of correct predictions
                    correct = (logits.argmax(dim=-1) == label).sum().item()
                    results["correct"] += correct#正确数
                    results["n"] += len(label)#总数

            dist.barrier()
            gathered_data = [None for _ in range(world_size)]#聚集各GPU结果
            dist.all_gather_object(gathered_data, results)

            if rank == 0:
                combined_data = {"correct": 0, "n": 0}#初始化结果字典
                for gpu_data in gathered_data:
                    combined_data["correct"] += gpu_data["correct"]
                    combined_data["n"] += gpu_data["n"]
                top1 = combined_data['correct']/combined_data['n']#计算准确率
                print(f"Top1: {top1}")
                test_result[f"test/{dataset_name}"] = top1
                #保留两位小数
                epoch_stats[f"{dataset_name}"] = round(top1*100, 2)

        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)

        if local_rank == 0:
            stats_df.to_csv(os.path.join(cfg.log_dir, 'stats.tsv'), sep='\t')#导出

        wandb_log(test_result | {"test/Epoch": epoch + 1})

        if cfg.ckpt_dir is not None and epoch != -1 and local_rank == 0:
            model_path = os.path.join(cfg.ckpt_dir, f'checkpoint_{epoch}.pth')
            clip_encoder_s.module.save(model_path)#保存模型
            
            optim_path = os.path.join(cfg.ckpt_dir, f'optimizer_{epoch}.pth')
            torch.save(optimizer.state_dict(), optim_path)#保存优化器状态

        dist.barrier()


        print(f"Epoch {epoch} finished")
        print(test_result)


if __name__ == "__main__":
    setup_ddp()
    # attachdebug()
    main_worker()
    cleanup_ddp()