from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
from src.models import utils

from src.datasets import ImageNet

# def eval_single_dataset(image_classifier, dataset, args, classification_head):

#     model = image_classifier
#     input_key = 'images'
#     image_enc = None

#     model.eval()
#     classification_head.eval()

#     dataloader = get_dataloader(dataset,
#                                 is_train=False,
#                                 args=args,
#                                 image_encoder=image_enc)

#     batched_data = enumerate(dataloader)
#     device = args.device

#     if hasattr(dataset, 'post_loop_metrics'):
#         # keep track of labels, predictions and metadata
#         all_labels, all_preds, all_metadata = [], [], []

#     with torch.no_grad():
#         top1, correct, n = 0., 0., 0.
#         for i, data in batched_data:

#             data = maybe_dictionarize(data)
#             x = data[input_key].to(device)
#             y = data['labels'].to(device)

#             if 'image_paths' in data:
#                 image_paths = data['image_paths']

#             logits = utils.get_logits(x, model, classification_head)

#             projection_fn = getattr(dataset, 'project_logits', None)
#             if projection_fn is not None:
#                 logits = projection_fn(logits, device)

#             if hasattr(dataset, 'project_labels'):
#                 y = dataset.project_labels(y, device)
#             pred = logits.argmax(dim=1, keepdim=True).to(device)
#             if hasattr(dataset, 'accuracy'):
#                 acc1, num_total = dataset.accuracy(logits, y, image_paths,
#                                                    args)
#                 correct += acc1
#                 n += num_total
#             else:
#                 correct += pred.eq(y.view_as(pred)).sum().item()
#                 n += y.size(0)

#             if hasattr(dataset, 'post_loop_metrics'):
#                 all_labels.append(y.cpu().clone().detach())
#                 all_preds.append(logits.cpu().clone().detach())
#                 metadata = data[
#                     'metadata'] if 'metadata' in data else image_paths
#                 all_metadata.extend(metadata)

#         top1 = correct / n

#         if hasattr(dataset, 'post_loop_metrics'):
#             all_labels = torch.cat(all_labels)
#             all_preds = torch.cat(all_preds)
#             metrics = dataset.post_loop_metrics(all_labels, all_preds,
#                                                 all_metadata, args)
#             if 'acc' in metrics:
#                 metrics['top1'] = metrics['acc']
#         else:
#             metrics = {}
#     if 'top1' not in metrics:
#         metrics['top1'] = top1

#     return metrics



# def evaluate(image_classifier, args):

#     for i, dataset_name in enumerate(args.eval_datasets):
#         dataset_class = getattr(datasets, dataset_name)
#         dataset = dataset_class(image_classifier.module.val_preprocess,
#                                     location=args.data_location,
#                                 batch_size=args.batch_size)
#         results = eval_single_dataset(image_classifier, dataset, args,
#                                       classification_head)


#     args.current_epoch = epoch
#     classification_head_new = get_zeroshot_classifier(args, model.module.model)
#     classification_head_new = classification_head_new.cuda()
#     eval_results = evaluate(model, args, classification_head_new, epoch_stats, logger)


def eval_single_dataset(image_classifier, dataset, args, classification_head):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics

if __name__ == "__main__":
    args = parse_arguments()
    cket_path = f"checkpoints/ImageNet/flyp_loss/{args.model}_BS512_WD0.1_LR1e-05_run1/checkpoint_9.pt"
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    clip_encoder_ft = CLIPEncoder(args, keep_lang=True)
    classification_head = ClassificationHead(normalize=True, weights=None)
    clip_encoder_ft.load(cket_path)
    stat_dict = torch.load(cket_path, weights_only=True)
    clip_encoder.cuda()
    clip_encoder_ft.cuda()

    classification_head = get_zeroshot_classifier(args, clip_encoder.model)
    classification_head_ft = get_zeroshot_classifier(args, clip_encoder_ft.model)

    from PIL import Image
    img_path = "datasets/data/ILSVRC2012/val/n01440764/ILSVRC2012_val_00000293.JPEG"
    image = Image.open(img_path)

    dataset_class = getattr(datasets, "ImageNet")

    
    dataset = dataset_class(clip_encoder.val_preprocess,
                                location=args.data_location,
                                batch_size=args.batch_size)

    print("-----------------zero shot------------------")
    # img = clip_encoder.val_preprocess(image).unsqueeze(0).cuda()
    # logits = utils.get_logits(img, clip_encoder, classification_head)
    # max_values, max_indices = torch.max(logits, dim=1)
    # print(max_values, max_indices)

    results = eval_single_dataset(clip_encoder, dataset, args, classification_head)

    print(results)



    print("-----------------fine tune------------------")
    # img = clip_encoder_ft.val_preprocess(image).unsqueeze(0).cuda()
    # logits = utils.get_logits(img, clip_encoder_ft, classification_head_ft)
    # max_values, max_indices = torch.max(logits, dim=1)
    # print(max_values, max_indices)



    
    results = eval_single_dataset(clip_encoder_ft, dataset, args, classification_head_ft)


    print(results)
