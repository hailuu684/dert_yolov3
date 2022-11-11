import torch

from ..models.matcher import *
from ..models.criterion import accuracy
from ..box_ops import get_world_size, is_dist_avail_and_initialized
from ..ultis.ultis import num_classes
from ..yolov3_models.ultis import bbox_x1y1x2y2_to_cxcywh

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


def find_divisible_number(tensor: Tensor, number: int):
    """
    find a divisible number of a tensor
    :param number: integer number
    :param tensor: shape = (..., ...)
    :return: divisible number of a first dimension
    """
    first_dimension = tensor.size(0)
    second_dimension = tensor.size(1)
    divisible_num = first_dimension
    # for _ in range(100):
    #     if first_dimension % number != 0:
    #         first_dimension += 1
    #     else:
    #         break
    while divisible_num % number != 0:
        divisible_num += 1
    return first_dimension, second_dimension, divisible_num


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

                    # Zero out all the detections that have IoU > threshold
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
        # print('out shape', output.size())  # (56534, 8)
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
        filtered_out = self.process_yolov3_pred(out, confidence=0.5, nms_conf=0.5)  # shape = (56534,8)

        ## Get lables padded
        filtered_out_clone = torch.clone(filtered_out)

        # Get class score and obj score
        cls_obj = filtered_out_clone[:, 6:]  # (..., 8)
        cls_score, ind = cls_obj[:, 0], cls_obj[:, 1].int()
        size_ = num_classes  # max(ind) + 1
        cls_clone = torch.zeros(size=(cls_score.size(0), size_))

        # (cls score: 0.4, index: 3) --> [0 0 0 0.4]
        for i in range(cls_clone.size(0)):
            # print(ind[i])
            cls_clone[i][ind[i]] = cls_score[i]

        out_bbx = filtered_out[:, 1:5]  # filtered_out[:, 0] is image_index
        out_cls = cls_clone  # shape = (3763, num_class)

        # add 1e-5 to avoid zero problems
        out_cls = torch.where(out_cls != 0, out_cls, 1e-5)

        # TODO: add 1 more dimension, this dimension should contain 0, but this is only for testing dimension
        #  compatibility
        add_out_cls = torch.zeros(size=(out_cls.size(0), 1))
        add_out_cls = torch.where(add_out_cls == 0,
                                  5.0,
                                  add_out_cls)

        # (..., num_class + 1)
        out_cls = torch.cat((out_cls, add_out_cls), 1)

        # Reshape: (..., ...) --> (batch_size, ..., num_class+1 for out_cls, 4 for out_bbx)
        out_cls, out_bbx = self.reshape_output(out_cls, out_bbx)

        # Convert bbx to cxcywh format
        out_bbx = bbox_x1y1x2y2_to_cxcywh(out_bbx)

        # normalize bbx
        out_bbx = out_bbx / 416.0  # image_size = (416,416)

        return {'pred_boxes': out_bbx.cuda(), 'pred_logits': out_cls.cuda()}

    def reshape_output(self, output_cls, output_bbx):
        """
        After filter out the outputs from yolov3, it is necessary to convert to the correct shapes
        that requires by dert's matcher.
        :param output_cls: (...,92)
        :param output_bbx: (...,4)
        :return: output_cls: (batch_size,...,92), output_bbx: (batch_size,...,4)
        """
        # Find a divisible number
        first_dim, num_cls, division_num_cls = find_divisible_number(output_cls, batch_size)
        _, bbx_dim, division_num_bbx = find_divisible_number(output_bbx, batch_size)

        if division_num_cls > first_dim:
            dummy_tensor_size = division_num_cls - first_dim

            # create dummy tensor
            dummy_tensor_cls = torch.zeros((dummy_tensor_size, num_cls))
            dummy_tensor_bbx = torch.zeros((dummy_tensor_size, bbx_dim))

            # concat to outputs
            output_cls = torch.cat((output_cls.cuda(), dummy_tensor_cls.cuda()), 0)
            output_bbx = torch.cat((output_bbx.cuda(), dummy_tensor_bbx.cuda()), 0)

        # Reshape
        output_cls = output_cls.view(batch_size, -1, num_cls)
        output_bbx = output_bbx.view(batch_size, -1, bbx_dim)
        return output_cls, output_bbx

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
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
        # Process outputs
        outputs = self.pre_process_pred(outputs)

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
