import torch

from .models.backbone import build_backbone
from .models.matcher import build_matcher
from .models.criterion import SetCriterion
from .models.transformer import build_transformer
from .models.dert import DETR
from .coco_dataset import build_dataset_train, build_yolov3_dataset, build_yolov3_dataset_500
from .ultis.import_packages import *

# Import functions from yolov3
from .yolov3_models.test import test_forward_once, dummy_process_output
from .yolov3_models.models import build_yolov3_model, freeze_model, build_dummy_yolov3_model
from .yolov3_models.criterion_yolov3 import SetCriterion_Yolov3
import os

# Import engine for training
from .engine import train_one_epoch, train_one_epoch_debug

# Import torch summary
from torchsummary import summary
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
    yolov3 = build_yolov3_model(coco_dataset=True, load_weights=True)
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
    matcher = build_matcher(set_cost_class=set_cost_class,
                            set_cost_bbox=set_cost_bbox,
                            set_cost_giou=set_cost_giou)

    # Build criterion
    # TODO: test aux_output
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

    # Freeze model
    yolov3, optimizer = freeze_model(yolov3, weight_decay)

    # Start training
    for epoch in range(num_epochs):  # TODO: change num_class in cfg file if needed
        # train_stats = train_one_epoch(yolov3, criterion, dataloader,
        #                               optimizer, device, epoch)

        # debug
        train_stats = train_one_epoch_debug(yolov3, criterion, dataloader,
                                            optimizer, device, epoch, aux=True)

        lr_scheduler.step()


def summary_dert_model():

    # print('Loading dataset.........')
    # dataset, data_loader_train = build_dataset_train()

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

    from transformers import DetrConfig, DetrForObjectDetection
    config = DetrConfig(use_pretrained_backbone=False)
    model = DetrForObjectDetection(config)
    #  Next, can try use that to train.
    #
    # Summary model
    print(summary(model.cuda(),
                  input_size=(3, 416, 416), batch_size=batch_size)) # TODO: Error on torchsummary


def summary_model_yolov3():
    """
    Summary yolov3 model
    :return: statistic table of the yolov3 model
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build yolov3 model
    yolov3 = build_dummy_yolov3_model(coco_dataset=True, load_weights=True, cuda=device)
    # ------------------ Before freezing ---------------------
    # Total params: 62,008,384
    # Trainable params: 62,008,384
    # Non-trainable params: 0
    # Input size: 31.69
    # Forward/ backward size: 14209.86
    # Params size: 236 MB
    # Estimated Total size: 14478 MB
    # ---------------------------------------------------------

    # Freeze model
    yolov3, optimizer = freeze_model(yolov3, weight_decay)
    # ------------------ After freezing ---------------------
    # Total params: 62,008,384
    # Trainable params: 561,960
    # Non-trainable params: 61,491,09
    # Input size: 31.69
    # Forward/ backward size: 14209.86
    # Params size: 236 MB
    # Estimated Total size: 14478 MB
    # ---------------------------------------------------------

    # Summary model
    print(summary(yolov3.cuda(),
                  input_size=(3, 416, 416), batch_size=batch_size))


def test_load_500_images():
    dataset, dataloader = build_yolov3_dataset_500()
    print(len(dataset))
    print(len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for count, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if count == 1:
            print(targets)
            break


if __name__ == "__main__":
    # dert_model()
    # train_yolov3_dert_loss() # TODO: Reduce the training time, class_error always 100
    # summary_model_yolov3()
    summary_dert_model()
    # test_load_500_images()