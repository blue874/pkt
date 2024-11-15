import os
import torch
from tqdm import tqdm

import numpy as np

import clip.clip as clip

import src.templates as templates
import src.datasets as datasets

from src.args import parse_arguments
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.eval import evaluate
from torch.utils.data import Dataset, DistributedSampler, DataLoader

import torch.distributed as dist

class promptSet(Dataset):
    def __init__(self, classname, template) -> None:
        super().__init__()
        self.classname = classname
        self.template = template
        self.prompts =[]
        self.label = []
        for i, class_name in enumerate(classname):
            for t in template:
                self.prompts.append(t(class_name))
                self.label.append(i)
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return clip.tokenize(self.prompts[idx]).squeeze(0), self.label[idx]


def get_zeroshot_classifier(args, clip_model, dataset=None):
    '''
    fixme: Some implementation details are missing, such as dataset
    Get a zero-shot classifier for the given dataset
    args: arguments
    clip_model: CLIP model
    dataset: dataset name
    '''
    rank = int(os.environ['RANK']) 
    world_size = int(os.environ['WORLD_SIZE'])
    assert args.model.template is not None
    assert dataset is not None
    template = getattr(templates, args.model.template)
    logit_scale = clip_model.logit_scale

    few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
    dataset_class = getattr(datasets, dataset)
    if dataset in few_shot_data_list:
        # assert args.k != None
        # assert args.k != 0
        print(f"Doing {args.k} shot classification")
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size_per_gpu,
                                k=args.k)
    else:
        dataset = dataset_class(None,
                                location=args.data_location,
                                batch_size=args.batch_size_per_gpu)
    
    num_classes = len(dataset.classnames)
    embedding_dim = clip_model.token_embedding.embedding_dim
    promptset = promptSet(dataset.classnames, template)
    del dataset
    sampler = DistributedSampler(promptset, shuffle=False, num_replicas=world_size, rank=rank)
    prompt_loader = DataLoader(promptset, batch_size=args.eval_batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id))

    dist.barrier()
    sum_embeddings = [torch.zeros(embedding_dim, device='cuda') for _ in range(num_classes)]
    counts = [torch.zeros(1, device='cuda', dtype=torch.float32) for _ in range(num_classes)]

    with torch.no_grad():
        for i, (texts, labels) in enumerate(tqdm(prompt_loader, total=len(prompt_loader), desc=f"Rank {rank} Processing Dataset", position=rank)):
            texts = texts.cuda()
            labels = labels.cuda()
            unique_labels = labels.unique()
            embeddings = clip_model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            for lb in unique_labels:
                mask = labels == lb
                selected_embeddings = embeddings[mask]
                sum_embeddings[lb] += selected_embeddings.sum(dim=0)
                counts[lb] += selected_embeddings.shape[0]
    
    sum_embeddings_tensor = torch.stack(sum_embeddings, dim=0)
    counts_tensor = torch.stack(counts, dim=0).squeeze(-1)

    dist.all_reduce(sum_embeddings_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

    mean_embeddings = sum_embeddings_tensor / counts_tensor.unsqueeze(1)  
    mean_embeddings = mean_embeddings / mean_embeddings.norm(dim=-1, keepdim=True) # N, D

    mean_embeddings = mean_embeddings.unsqueeze(1) # D, N, 1
    mean_embeddings = torch.transpose(mean_embeddings, 0, 2)

    mean_embeddings *= logit_scale.exp()

    mean_embeddings = mean_embeddings.squeeze().float()
    mean_embeddings = torch.transpose(mean_embeddings, 0, 1)

    classification_head = ClassificationHead(normalize=True,
                                             weights=mean_embeddings)
    return classification_head



# def get_zeroshot_classifier_bycsv(args,clip_model,datapath=None):
#     '''
#     fixme: Some implementation details are missing, such as dataset
#     Get a zero-shot classifier for the given dataset
#     args: arguments
#     clip_model: CLIP model
#     dataset: dataset name
#     '''
#     rank = int(os.environ['RANK']) 
#     world_size = int(os.environ['WORLD_SIZE'])
#     assert args.model.template is not None
#     assert dataset is not None
#     template = getattr(templates, args.model.template)
#     logit_scale = clip_model.logit_scale

#     few_shot_data_list = ["ImageNetKShot", "PatchCamelyonVal"]
#     dataset_class = getattr(datasets, dataset)
#     if dataset in few_shot_data_list:
#         # assert args.k != None
#         # assert args.k != 0
#         print(f"Doing {args.k} shot classification")
#         dataset = dataset_class(None,
#                                 location=args.data_location,
#                                 batch_size=args.batch_size_per_gpu,
#                                 k=args.k)
#     else:
#         dataset = dataset_class(None,
#                                 location=args.data_location,
#                                 batch_size=args.batch_size_per_gpu)
    
#     num_classes = len(dataset.classnames)
#     embedding_dim = clip_model.token_embedding.embedding_dim
#     promptset = promptSet(dataset.classnames, template)
#     del dataset
#     sampler = DistributedSampler(promptset, shuffle=False, num_replicas=world_size, rank=rank)
#     prompt_loader = DataLoader(promptset, batch_size=args.eval_batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id))

#     dist.barrier()
#     sum_embeddings = [torch.zeros(embedding_dim, device='cuda') for _ in range(num_classes)]
#     counts = [torch.zeros(1, device='cuda', dtype=torch.float32) for _ in range(num_classes)]

#     with torch.no_grad():
#         for i, (texts, labels) in enumerate(tqdm(prompt_loader, total=len(prompt_loader), desc=f"Rank {rank} Processing Dataset", position=rank)):
#             texts = texts.cuda()
#             labels = labels.cuda()
#             unique_labels = labels.unique()
#             embeddings = clip_model.encode_text(texts)
#             embeddings /= embeddings.norm(dim=-1, keepdim=True)

#             for lb in unique_labels:
#                 mask = labels == lb
#                 selected_embeddings = embeddings[mask]
#                 sum_embeddings[lb] += selected_embeddings.sum(dim=0)
#                 counts[lb] += selected_embeddings.shape[0]
    
#     sum_embeddings_tensor = torch.stack(sum_embeddings, dim=0)
#     counts_tensor = torch.stack(counts, dim=0).squeeze(-1)

#     dist.all_reduce(sum_embeddings_tensor, op=dist.ReduceOp.SUM)
#     dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

#     mean_embeddings = sum_embeddings_tensor / counts_tensor.unsqueeze(1)  
#     mean_embeddings = mean_embeddings / mean_embeddings.norm(dim=-1, keepdim=True) # N, D

#     mean_embeddings = mean_embeddings.unsqueeze(1) # D, N, 1
#     mean_embeddings = torch.transpose(mean_embeddings, 0, 2)

#     mean_embeddings *= logit_scale.exp()

#     mean_embeddings = mean_embeddings.squeeze().float()
#     mean_embeddings = torch.transpose(mean_embeddings, 0, 1)

#     classification_head = ClassificationHead(normalize=True,
#                                              weights=mean_embeddings)
#     return classification_head

def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args,
                                                      image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder,
                                     classification_head,
                                     process_images=False)

    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)