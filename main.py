import torch

from .models.backbone import build_backbone
from .models.matcher import build_matcher
from .models.criterion import SetCriterion
from .models.transformer import build_transformer
from .models.dert import DETR
from .coco_dataset import build_dataset_train, build_yolov3_dataset
from .ultis.import_packages import *

# Import functions from yolov3
from .yolov3_models.test import test_forward_once, dummy_process_output
from .yolov3_models.models import build_yolov3_model
from .yolov3_models.criterion_yolov3 import SetCriterion_Yolov3
import os

# Import engine for training
from .engine import train_one_epoch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def dert_model():
    """
    input_target_shape = list[dict0(), dict1(), dict2(),...]. len() == batch_size if dataloader
                            {'bboxes': (num_objects, 4)
                            'labels': (num_objects),
                            'id': (num_objects),
                            'area':(num_objects),
                            'iscrowed': (num_objects)}

    output_class shape = (batch_size, num_queries, num_class + 1)
    output_coord_shape = (batch_size, num_queries, 4)
    """

    print('Loading dataset.........')
    dataset, data_loader_train = build_dataset_train()

    print('Loading backbone.........')
    backbone = build_backbone()

    # Build transformer
    transformer = build_transformer(hidden_dim=hidden_dim, dropout=dropout,
                                    nheads=nheads, dim_feedforward=dim_feedforward,
                                    enc_layers=enc_layers, dec_layers=dec_layers,
                                    pre_norm=False)

    # Build DERT
    dert_model = DETR(backbone=backbone, transformer=transformer, num_classes=num_classes,
                      num_queries=num_queries, aux_loss=False)

    # Get parameter dicts
    model_without_ddp = dert_model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr,
        },
    ]

    # Build matcher
    matcher = build_matcher(set_cost_class=0.2,
                            set_cost_bbox=0.2,
                            set_cost_giou=0.2)

    # Build criterion
    criterion = SetCriterion(num_classes=num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=eos_coef,
                             losses=losses)

    # Send the weights to the GPU
    # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte?rq=1
    if torch.cuda.is_available():
        dert_model.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for count, (samples, targets) in enumerate(data_loader_train):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if count == 1:
            # # features, pos = backbone(samples)
            # # src, mask = features[-1].decompose()
            # outputs = dert_model(samples)
            #
            # outputs_bbx = outputs['pred_boxes']
            # outputs_class = outputs['pred_logits']
            # print(f'output_box shape = {outputs_bbx.size()}')
            # print(f'outputs_class shape = {outputs_class.size()}')
            # # Compute loss
            # loss = criterion(outputs, targets)
            # print(targets)

            # # Copy the targets --> test the loss function with outputs from yolov3
            yolov3_target = targets.copy()
            break

    # # Test criterion with yolov3 architecture

    # ----------------------------------------------------------------
    # ----------------- TEST CRITERION WITH YOLOV3 OUTPUTS -----------
    # ----------------------------------------------------------------
    # # Test compatibility
    # Test: succeeded
    yolov3_target = dummy_process_output(yolov3_target)
    yolov3_loss = test_forward_once(yolov3_target, matcher, weight_dict, eos_coef, losses)
    print(yolov3_loss)
    # -------------------------------------------------------------------------------


def train_yolov3_dert_loss():
    # mAP: https://github.com/facebookresearch/detr/blob/main/util/plot_utils.py
    # Build dataset
    dataset, dataloader = build_yolov3_dataset()

    # Build yolov3 model
    yolov3 = build_yolov3_model(coco_dataset=True, load_weights=False)
    # TODO: remember to check batch_size in config file

    # Get parameter dicts
    model_without_ddp = yolov3
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr,
        },
    ]

    # Use GPU
    if torch.cuda.is_available():
        yolov3.cuda()

    # Build matcher
    matcher = build_matcher(set_cost_class=0.2,
                            set_cost_bbox=0.2,
                            set_cost_giou=0.2)

    # Build criterion
    criterion = SetCriterion_Yolov3(num_classes=num_classes,
                                    matcher=matcher,
                                    weight_dict=weight_dict,
                                    eos_coef=eos_coef,
                                    losses=losses)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define number of epochs
    num_epochs = 5

    # Optimizer
    optimizer = optim.AdamW(param_dicts, lr=lr,
                            weight_decay=weight_decay)

    # Lr scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    # Start training
    for epoch in range(num_epochs):  # TODO: change num_class in cfg file if needed
        train_stats = train_one_epoch(yolov3, criterion, dataloader,
                                      optimizer, device, epoch)

        lr_scheduler.step()

if __name__ == "__main__":
    # dert_model()
    train_yolov3_dert_loss()
    # test()
