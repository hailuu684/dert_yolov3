from .models import build_yolov3_model
from .ultis import process_image_yolov3
import torch
from .criterion_yolov3 import SetCriterion_Yolov3


def test_forward_once(yolov3_target, matcher, weight_dict, eos_coef, losses):
    """
    :param yolov3_target: max(yolov3_target['labels]) <= num_class defined
    :param matcher: matcher creation
    :param weight_dict: weight_dict creation
    :param eos_coef: eos_coef creation
    :param losses: losses creation
    :return: loss

    The output from yolov3 is (batch, 10647, 5 + num_class). In this test case, batch_size = 1
    because there is only 1 image is read --> make sure set batch_size = 1 at ultils.import_packages
    and num_queries = 10647
    """
    yolov3_module = build_yolov3_model()
    if torch.cuda.is_available():
        yolov3_module.cuda()
    # example image: /media/luu/coco/train2017/000000000009.jpg
    example_image_path = '/media/luu/coco/train2017/000000000009.jpg'

    # Output from yolov3 module          (batch, 10647, 5 + num_class)
    processed_image = process_image_yolov3(example_image_path)  # shape = (batch, 3, h, w)

    # Make prediction
    yolov3_output = yolov3_module(processed_image.cuda(), torch.cuda.is_available())

    # yolo_pred_bbx
    out_bbx = yolov3_output[:, :, :4]

    # yolo_pred_cls
    out_cls = yolov3_output[:, :, 5:]

    # yolo_pred_objectness
    out_obj = yolov3_output[:, :, 5, None]  # None means creating 1 more dimension

    # add to dict to feed to dert's criterion
    dict_out = {'pred_logits': out_cls.cuda(), 'pred_boxes': out_bbx.cuda(),
                'pred_objectness': out_obj.cuda()}

    # Set up criterion
    yolov3_criterion = SetCriterion_Yolov3(num_classes=4,
                                           matcher=matcher,
                                           weight_dict=weight_dict,
                                           eos_coef=eos_coef,
                                           losses=losses)

    yolov3_loss = yolov3_criterion(dict_out, yolov3_target)
    return yolov3_loss
    # print(yolov3_loss)