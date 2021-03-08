import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss


def add_rand_attr_sample(ignore_mask):
    """
    ignore_mask: (num_RoI,) fp32 [0, MAX_ATTR_PER_OBJ]
    """
    positive_mask = ignore_mask.clamp(0, 1)
    negative_mask = 1 - positive_mask
    num_pos = max(positive_mask.sum() // 4, 1)
    k = int(min(ignore_mask.shape[0] - num_pos, num_pos))
    _, indices = torch.topk(negative_mask, k, dim=0)
    ignore_mask[indices] = 1.0
    return ignore_mask


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    # loss = F.binary_cross_entropy(F.softmax(pred, dim=-1), target, reduction='none')
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    # neg_mask = (target <= 0.5).float()
    # loss = loss * (1 - neg_mask) * 2 + 0.5 * loss * neg_mask

    ignore_mask = (target.sum(-1, keepdim=True) > 0).float()  # ignore proposal with no attribute label
    # ignore_mask = add_rand_attr_sample(ignore_mask)
    loss = (loss * focal_weight * ignore_mask).sum(dim=-1)
    loss = weight_reduce_loss(loss, weight, reduction, ignore_mask.sum())

    # if float(loss) < 0.0:
    #     print(pred, pred.min(), pred.max())
    #     print(target, target.min(), target.max())
    #     import pdb; pdb.set_trace()
    #     print('-' * 100)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # loss_cls = self.loss_weight * sigmoid_focal_loss(
            #     pred,
            #     target,
            #     weight,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     reduction=reduction,
            #     avg_factor=avg_factor)
            
            if pred.dim() == target.dim() and pred.shape[-1] != target.shape[-1] and target.dtype == torch.long:
                num_class = pred.shape[-1]
                num_attr_per_obj = float(target.shape[-1])

                onehot = torch.nn.functional.one_hot(target, num_classes=num_class).float()
                onehot = onehot.sum(dim=1) # [..., :-1]  # remove background/no-attr class
                bg = onehot[..., -1:]
                bg = bg * (bg > num_attr_per_obj - 1e-5).float()
                bg = bg.clamp(0, 1)
                target = torch.cat([onehot[..., :-1], bg], dim=-1)

                pred = pred[..., :-1]
                target = target[..., :-1]
            
            loss_cls = py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
