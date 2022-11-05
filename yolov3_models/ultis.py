import torch

from .initialize_packages import *
import cv2


def process_image_yolov3(path):
    image = cv2.imread(path)
    img = cv2.resize(image, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


def bbox_x1y1x2y2_to_xywh(box):
    bx, by = box[..., 0], box[..., 1]
    bw = box[..., 2] - box[..., 0]
    bh = box[..., 3] - box[..., 1]
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = bx, by, bw, bh
    return box


def bbox_x1y1x2y2_to_cxcywh(box):
    bw = box[..., 2] - box[..., 0]
    bh = box[..., 3] - box[..., 1]
    cx, cy = box[..., 0] + bw / 2, box[..., 1] + bh / 2
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = cx, cy, bw, bh
    return box


def bbox_cxcywh_to_x1y1x2y2(box):
    x1, x2 = box[..., 0] - box[..., 2] / 2, box[..., 0] + box[..., 2] / 2
    y1, y2 = box[..., 1] - box[..., 3] / 2, box[..., 1] + box[..., 3] / 2
    box[..., 0], box[..., 1], box[..., 2], box[..., 3] = x1, y1, x2, y2
    return box


def bbox_cxcywh_to_xywh(box):
    x, y = box[..., 0] - box[..., 2] / 2, box[..., 1] - box[..., 3] / 2
    box[..., 0], box[..., 1] = x, y
    return box


# Convert Coco bb to Yolo
def coco_to_yolo(box, image_w, image_h):
    clone_box = torch.clone(box)
    clone_box[..., 0] = ((2 * box[..., 0] + box[..., 2]) / (2 * image_w))
    clone_box[..., 1] = ((2 * box[..., 1] + box[..., 3]) / (2 * image_h))
    clone_box[..., 2] = box[..., 2] / image_w
    clone_box[..., 3] = box[..., 3] / image_h
    return clone_box

