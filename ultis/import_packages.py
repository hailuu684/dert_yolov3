import torch
import torchvision
# from fastai import *
# from fastai.vision.all import *
# from fastai.imports import *
# import cv2 # not installed yet
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchvision.ops.boxes import box_area
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
import torchvision.transforms.functional as F
import PIL
import random
from typing import Optional, List

import glob
import PIL

# Github tokens: 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 2nd/ Nov/ 2022
num_queries = 10647 # 10647: (batch, 10647, 8) output from yolov3 works
num_classes = 4  # COCO: 91         Custom: 4
batch_size = 5
weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 0.5}
eos_coef = 0.2
losses = ['labels', 'boxes', 'cardinality']

# Initialize backbone
dropout = 0.1
nheads = 8
dim_feedforward = 2048
hidden_dim = 256
enc_layers = 6
dec_layers = 6
pre_norm = False  # choose between True or False

# Path
train_path = '/media/luu/coco/train2017'
train_anno_path = '/media/luu/coco/annotations/instances_train2017.json'
