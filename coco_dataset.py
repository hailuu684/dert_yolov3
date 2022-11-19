import torchvision.transforms.functional

from .transforms import *
from .ultis.ultis import collate_fn
from .yolov3_models.ultis import *
from .ultis.import_packages import num_worker
# Coco Format: [x_min, y_min, width, height]
# Pascal_VOC Format: [x_min, y_min, x_max, y_max]
# Darknet label format: [label_index, cx, cy, w, h] (Relative coordinates)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class CoCo_YOLOv3(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CoCo_YOLOv3, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, targets = super(CoCo_YOLOv3, self).__getitem__(idx)

        # # resize image --> fit to yolov3's input
        # img = img.resize((416, 416))

        # Get index and annotations
        image_id = self.ids[idx]
        targets = {'image_id': image_id, 'annotations': targets}
        img, targets = self.prepare(img, targets)

        # # If transform is None, image should be converted to tensor --> default collate_fn will work
        # img = torchvision.transforms.functional.to_tensor(img)
        if self._transforms is not None:
            img, targets = self._transforms(img, targets)

        # -------------------------------------------------------------------------------
        # ----------------- Do we really need to convert bbx to yolo format? ------------
        # -------------------------------------------------------------------------------
        # format may not important in this case since we are using dert loss
        # # Un-normalize bbox
        # targets['boxes'] = un_normalize_bbx(targets['boxes'],
        #                                     targets['size'][0], targets['size'][1])

        # # Convert coco format --> yolo format
        # targets['boxes'] = coco_to_yolo(targets['boxes'],
        #                                 targets['size'][0], targets['size'][1])
        return img, targets


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:  # For now don't need segmentation
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize(scales, max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_yolo_transform(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            Resize([416, 416]),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset_train():
    # Make dataset
    dataset = CocoDetection(train_path, train_anno_path,
                            transforms=make_coco_transforms('train'),
                            return_masks=False)

    # Make dataloader
    data_loader_train = DataLoader(dataset, batch_size=batch_size,
                                   collate_fn=collate_fn, num_workers=0)

    return dataset, data_loader_train


def build_yolov3_dataset():
    dataset = CoCo_YOLOv3(train_path, train_anno_path,
                          transforms=make_yolo_transform('train'), return_masks=False)

    data_loader = DataLoader(dataset, batch_size, num_workers=num_worker,
                             collate_fn=collate_fn_yolov3)
    return dataset, data_loader


def build_yolov3_dataset_500():
    dataset = CoCo_YOLOv3(train_path, train_anno_path,
                          transforms=make_yolo_transform('train'), return_masks=False)

    # Total dataset = 118287
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               (len(dataset)-500, 500))
    data_loader = DataLoader(val_dataset, batch_size, num_workers=num_worker,
                             collate_fn=collate_fn_yolov3)
    return val_dataset, data_loader
