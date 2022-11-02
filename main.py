import torch

from .models.backbone import build_backbone
from .models.matcher import build_matcher
from .models.criterion import SetCriterion
from .models.transformer import build_transformer
from .models.dert import DETR
from .coco_dataset import build_dataset_train
from .ultis.import_packages import *
from .yolov3_models.models import build_yolov3_model
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
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
            # print(loss)

            # # Copy the targets --> test the loss function with outputs from yolov3
            # yolov3_target = targets.copy()
            break

    # Test: succeeded
    # yolov3_target = dummy_process_output(yolov3_target)
    # # Test criterion with yolov3 architecture

    # ----------------------------------------------------------------
    # ----------------- TEST CRITERION WITH YOLOV3 OUTPUTS -----------
    # ----------------------------------------------------------------
    # Create yolov3
    yolov3_module = build_yolov3_model()

    # Get image from coco dataset
    image, _ = dataset.__getitem__(20)

    # import function
    from .yolov3_models.ultis import process_image_yolov3

    # Output from yolov3 module          (batch, 10647, 5 + num_class)
    processed_image = process_image_yolov3(image)  # shape = (batch, 3, h, w)

    # Make prediction
    yolov3_output = yolov3_module(processed_image, torch.cuda.is_available())

    # yolo_pred_bbx
    out_bbx = yolov3_output[:, :, :4]

    # yolo_pred_cls
    out_cls = yolov3_output[:, :, 5:]

    # yolo_pred_objectness
    out_obj = yolov3_output[:, :, 5, None]  # None means creating 1 more dimension

    # add to dict to feed to dert's criterion
    dict_out = {'pred_logits': out_cls.cuda(), 'pred_boxes': out_bbx.cuda(),
                'pred_objectness': out_obj}

    # yolov3 output
    # yolov3_out = {'pred_logits': yolov3_class.cuda(), 'pred_boxes': yolov3_bbx.cuda()}
    # yolov3_loss = criterion(yolov3_out, yolov3_target)
    # print(yolov3_loss)
    # print(yolov3_target)


def dummy_process_output(targets):
    for target in targets:
        target['labels'] = torch.where(target['labels'] > 5, 0, target['labels'])

    return targets


if __name__ == "__main__":
    main()