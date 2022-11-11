import torch

from .initialize_packages import *
import cv2
from typing import List
from ..ultis.import_packages import Tensor
from ..ultis.ultis import _max_by_axis, _onnx_nested_tensor_from_tensor_list
import torchvision


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


def un_normalize_bbx(box, image_w, image_h):
    box[..., 0] = box[..., 0] * image_w
    box[..., 1] = box[..., 1] * image_h
    box[..., 2] = box[..., 2] * image_w
    box[..., 3] = box[..., 3] * image_h
    return box


def collate_fn_yolov3(batch):
    batch = list(zip(*batch))
    batch[0] = tensor_from_tensor_list(batch[0])
    return tuple(batch)


def tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # print(max_size)
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return tensor
