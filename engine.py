from .models.coco_eval import CocoEvaluator

import math
import os
import sys
from typing import Iterable

from .ultis.misc import *
from .yolov3_models.test import dummy_process_output


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, device)
        # yolo_pred_bbx
        out_bbx = outputs[:, :, :4]

        # yolo_pred_cls
        out_cls = outputs[:, :, 5:]

        # yolo_pred_objectness
        out_obj = outputs[:, :, 5, None]  # None means creating 1 more dimension

        # add to dict to feed to dert's criterion
        dict_out = {'pred_logits': out_cls.cuda(), 'pred_boxes': out_bbx.cuda(),
                    'pred_objectness': out_obj.cuda()}

        # dict of losses
        loss_dict = criterion(dict_out, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Activate backpropagation
        losses.requires_grad = True

        # Loss backward
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_debug(model: torch.nn.Module, criterion: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, max_norm: float = 0, aux=False):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for index, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, device)  # shape = (batch, 10647, 5 + num_cls)
        # yolo_pred_bbx
        out_bbx = outputs[:, :, :4]

        # yolo_pred_cls
        out_cls = outputs[:, :, 5:]

        # yolo_pred_objectness
        out_obj = outputs[:, :, 5, None]  # None means creating 1 more dimension

        # aux_loss
        detection_1 = 507
        detection_2 = 2028
        detection_3 = 8112
        out_bbx_1 = outputs[:, :detection_1, :4].cuda()
        out_cls_1 = outputs[:, :detection_1, 5:].cuda()
        out_bbx_2 = outputs[:, detection_1:detection_2, :4].cuda()
        out_cls_2 = outputs[:, detection_1:detection_2, 5:].cuda()
        out_bbx_3 = outputs[:, detection_2:detection_3, :4].cuda()
        out_cls_3 = outputs[:, detection_2:detection_3, 5:].cuda()

        # aux_outputs
        aux_outputs = {'pred_logits': [out_cls_1, out_cls_2, out_cls_3],
                       'pred_boxes': [out_bbx_1, out_bbx_2, out_bbx_3]}

        if aux:
            dict_out = {'pred_logits': out_cls.cuda(), 'pred_boxes': out_bbx.cuda(),
                        'pred_objectness': out_obj.cuda(),
                        'aux_outputs': aux_outputs}
        else:
            # add to dict to feed to dert's criterion
            dict_out = {'pred_logits': out_cls.cuda(), 'pred_boxes': out_bbx.cuda(),
                        'pred_objectness': out_obj.cuda()}
        # dict of losses
        loss_dict = criterion(dict_out, targets)

        # print(loss_dict)
        if index == 0:
            break

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Activate backpropagation
        losses.requires_grad = True

        # Loss backward
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}