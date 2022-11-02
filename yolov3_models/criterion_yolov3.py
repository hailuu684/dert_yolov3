import torch

from ..models.matcher import *
from ..models.criterion import accuracy
from ..box_ops import get_world_size, is_dist_avail_and_initialized


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


class SetCriterion_Yolov3(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def process_yolov3_pred(self, outputs, confidence, nms_conf):
        conf_mask = (outputs[:, :, 4] > confidence).float().unsqueeze(2)
        prediction = outputs * conf_mask

        box_corner = prediction.new(prediction.shape)
        print('filtered boxes', box_corner.size())

        # Transform from cxcywh --> xyxy
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_corner[:, :, :4]

        batch_size = prediction.size(0)

        write = False

        for ind in range(batch_size):
            image_pred = prediction[ind]  # image Tensor
            # confidence threshholding
            # NMS

            max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            image_pred = torch.cat(seq, 1)

            non_zero_ind = (torch.nonzero(image_pred[:, 4]))
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
            except:
                continue

            if image_pred_.shape[0] == 0:
                continue
                #

            # Get the various classes detected in the image
            img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

            for cls in img_classes:
                # perform NMS

                # get the detections with one particular class
                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                # sort the detections such that the entry with the maximum objectness
                # confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)  # Number of detections

                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                    ind)  # Repeat the batch_id for as many detections of the class cls in the image
                seq = batch_ind, image_pred_class

                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))
        print('out shape', output.size())
        try:
            return output
        except:
            return 0

    def pre_process_pred(self, outputs):
        pred_bbx = outputs['pred_boxes']  # (batch, 10647, 4)
        pred_logits = outputs['pred_logits']  # (batch, 10647, 4)
        pred_obj = outputs['pred_objectness']  # (batch, 10647, 1)

        out = torch.cat((pred_bbx, pred_obj, pred_logits), 2)

        # reduce the number of not correct bounding boxes ( negative bbx, conf val < threshold)
        # filtered_out shape = (...,image_index + 4 corner coordinates + objectness score,
        #                      score of class + index of that class)
        filtered_out = self.process_yolov3_pred(out, confidence=0.5, nms_conf=0.5)

        # Need to convert to pred_logits = (batch, num_querries, num_class+1)
        #                    pred_bbx = (batch, num_querries, 4)

        ## Get lables padded
        filtered_out_clone = torch.clone(filtered_out)

        # Get class score and obj score
        cls_obj = filtered_out_clone[:, 6:]
        cls_score, ind = cls_obj[:, 0], cls_obj[:, 1].int()
        size_ = max(ind) + 1
        cls_clone = torch.zeros(size=(cls_score.size(0), size_))
        for i in range(cls_clone.size(0)):
            # print(ind[i])
            cls_clone[i][ind[i]] = cls_score[i]
            cls_clone[i] += torch.tensor([1e-5, 1e-5, 1e-5, 1e-5],
                                         dtype=torch.float64)
        out_bbx = filtered_out[:, 1:5]
        out_cls = cls_clone
        # TODO: out_cls still dont have correct shape, should be (batch, num_querries, num_cls+1)
        # TODO: add 1e-5 to cls_clone to avoid zero problems
        # TODO: Download git and push code
        return {'pred_boxes': out_bbx, 'pred_logits': out_cls}

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # Process outputs
        outputs = self.pre_process_pred(outputs)

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        # ------------------For custom dataset ---------------------------
        # idx = list(idx)
        # idx = [a.type(torch.int32) for a in idx]
        # idx = tuple(idx)
        #
        # # Works for custom dataset
        # target_classes[[a.type(torch.long) for a in list(idx)]] = target_classes_o.type(torch.long)
        # ----------------------------------------------------------------

        # Works for COCO dataset
        target_classes[idx] = target_classes_o

        src_logits_cuda = src_logits.transpose(1, 2).cuda()
        target_classes_cuda = target_classes.cuda()
        loss_ce = torch.nn.functional.cross_entropy(src_logits_cuda,
                                                    target_classes_cuda,
                                                    self.empty_weight.cuda())
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # Works for COCO dataset
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

            # ------------------For custom dataset ---------------------------
            # # Works for custom dataset
            # losses['class_error'] = 100 - accuracy(src_logits[[a.type(torch.long) for a in list(idx)]],
            #                                        target_classes_o.type(torch.long))[0]

            # ----------------------------------------------------------------

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # Process outputs
        outputs = self.pre_process_pred(outputs)

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = torch.nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # Process outputs
        outputs = self.pre_process_pred(outputs)

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
