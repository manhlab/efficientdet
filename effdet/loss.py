import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .anchors import decode_box_outputs

from typing import Optional, List

eps = 10e-16


def compute_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])

    inter = torch.where(min_x2 - max_x1 > 0, min_x2 - max_x1, torch.tensor(0.)) * \
            torch.where(min_y2 - max_y1 > 0, min_y2 - max_y1, torch.tensor(0.))
    union = area1 + area2 - inter
    iou = inter / union
    iou = torch.clamp(iou, min=0, max=1.0)
    return iou


def compute_g_iou(bboxes1, bboxes2):
    "box1 of shape [N, 4] and box2 of shape [N, 4]"
    # assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter
    C = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) * \
        (torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1]))
    g_iou = inter / union - (C - union) / C
    g_iou = torch.clamp(g_iou, min=0, max=1.0)
    return g_iou


def compute_d_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    # assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # squared euclidian distance between the target and predicted bboxes
    d_2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    # squared length of the diagonal of the minimum bbox that encloses both bboxes
    c_2 = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) ** 2 + (
            torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1])) ** 2
    d_iou = inter / union - d_2 / c_2
    d_iou = torch.clamp(d_iou, min=-1.0, max=1.0)

    return d_iou


def compute_c_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    # assert bboxes1.size(0) == bboxes2.size(0)
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])

    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    # squared euclidian distance between the target and predicted bboxes
    d_2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    # squared length of the diagonal of the minimum bbox that encloses both bboxes
    c_2 = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) ** 2 + (
            torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1])) ** 2
    iou = inter / union
    v = 4 / np.pi ** 2 * (torch.atan(w1 / h1) - torch.atan(w2 / h2)) ** 2
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v + eps)
    c_iou = iou - (d_2 / c_2 + alpha * v)
    c_iou = torch.clamp(c_iou, min=-1.0, max=1.0)
    return c_iou


def focal_loss(logits, targets, alpha: float, gamma: float, normalizer, num_cls, smooth_rate=0.1):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

        normalizer: A float32 scalar normalizes the total loss from all examples.

        num_cls: number of classes

        smooth_rate: A float32 number used for label smoothing

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """

    positive_label_mask = targets == 1.0
    targets = targets * (1 - smooth_rate) + smooth_rate/num_cls

    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


def huber_loss(
        input, target, delta: float = 1., weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def smooth_l1_loss(
        input, target, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = torch.abs(input - target)
        loss = torch.where(err < beta, 0.5 * err.pow(2) / beta, err - 0.5 * beta)
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def _classification_loss(cls_outputs, cls_targets, num_positives, num_cls, alpha: float = 0.25, gamma: float = 2.0):
    """Computes classification loss. Focal_loss"""
    normalizer = num_positives
    classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma, normalizer, num_cls + 1)
    return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_targets, box_outputs, weights=mask, delta=delta, size_average=False)
    box_loss /= normalizer
    return box_loss


class IouLoss(nn.Module):

    def __init__(self, losstype='Ciou', reduction='mean'):
        super(IouLoss, self).__init__()
        self.reduction = reduction
        self.loss = losstype

    def forward(self, target_bboxes, pred_bboxes):
        num = target_bboxes.shape[0]
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - compute_iou(target_bboxes, pred_bboxes))
        elif self.loss == 'Giou':
            loss = torch.sum(1.0 - compute_g_iou(target_bboxes, pred_bboxes))
        elif self.loss == 'Diou':
            loss = torch.sum(1.0 - compute_d_iou(target_bboxes, pred_bboxes))
        else:
            loss = torch.sum(1.0 - compute_c_iou(target_bboxes, pred_bboxes))

        if self.reduction == 'mean':
            return loss / num
        else:
            return loss


def one_hot(x, num_classes: int):
    # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
    x_non_neg = (x >= 0).to(x.dtype)
    onehot = torch.zeros(x.shape + (num_classes,), device=x.device, dtype=x.dtype)
    onehot.scatter_(-1, (x * x_non_neg).unsqueeze(-1), 1)
    return onehot * x_non_neg.unsqueeze(-1)


class DetectionLoss(nn.Module):
    def __init__(self, config, anchors, use_iou_loss=True):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight
        self.use_iou_loss = use_iou_loss
        if self.use_iou_loss:
            self.anchors = anchors
            self.iou_loss = IouLoss()

    def forward(
            self, cls_outputs: List[torch.Tensor], box_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor], box_targets: List[torch.Tensor], num_positives: torch.Tensor):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
                at each feature level (index)
            box_outputs: a List with values representing box regression targets in
                [batch_size, height, width, num_anchors * 4] at each feature level (index)
            cls_targets: groundtruth class targets.
            box_targets: groundtrusth box targets.
            num_positives: num positive grountruth anchors
        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.
            cls_loss: an integer tensor representing total class loss.
            box_loss: an integer tensor representing total box regression loss.
        """
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)

        cls_losses = []
        box_losses = []
        for l in range(levels):
            cls_targets_at_level = cls_targets[l]
            box_targets_at_level = box_targets[l]

            # Onehot encoding for classification labels.
            cls_targets_at_level_oh = one_hot(cls_targets_at_level, self.num_classes)

            bs, height, width, _, _ = cls_targets_at_level_oh.shape
            cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
            cls_loss = _classification_loss(
                cls_outputs[l].permute(0, 2, 3, 1),
                cls_targets_at_level_oh,
                num_positives_sum,
                alpha=self.alpha, gamma=self.gamma, num_cls=self.num_classes)
            cls_loss = cls_loss.view(bs, height, width, -1, self.num_classes)
            cls_loss *= (cls_targets_at_level != -2).unsqueeze(-1).float()
            cls_losses.append(cls_loss.sum())

            box_losses.append(_box_loss(
                box_outputs[l].permute(0, 2, 3, 1),
                box_targets_at_level,
                num_positives_sum,
                delta=self.delta))

        # Sum per level losses to total loss.
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss
