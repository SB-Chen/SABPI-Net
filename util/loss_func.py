from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from sklearn.metrics import r2_score
import numpy as np
import warnings
from typing import Optional
from torch import Tensor
from collections import defaultdict


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


def createLossFunc(config, epoch: Optional[int] = None):
    if hasattr(config, 'cls_num_list'):
        cls_num_list = config.cls_num_list
    else:
        cls_num_list = None

    per_cls_weights = None
    if config.train_rule == 'None':
        per_cls_weights = None
    elif config.train_rule == 'Resample':
        per_cls_weights = None
    elif config.train_rule == 'Reweight':
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(config.device)
    elif config.train_rule == 'DRW':
        train_sampler = None
        # idx = epoch // 60
        betas = [0, 0.9999]
        idx = min(epoch // 60, len(betas) - 1) 
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(config.device)
    else:
        warnings.warn('Sample rule is not listed')

    if config.loss_type == 'crossEntropyLoss' or config.loss_type == 'CE':
        criterion = CrossEntropyLoss(weight=per_cls_weights)
    elif config.loss_type == 'weightedCrossEntropyLoss':
        weight = torch.FloatTensor(config.classes_weight).to(config.device)
        criterion = CrossEntropyLoss(weight=weight)
    elif config.loss_type == 'ClassBalancedFocalLoss':
        criterion = ClassBalancedFocalLoss(cls_num_list=cls_num_list,
                                           gamma=1.0,
                                           reduction='mean',
                                           beta=0.9999
                                           )
    elif config.loss_type == 'Focal':
        assert per_cls_weights is not None, f"per_cls_weights is None"
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).to(config.device)
        criterion = InteractiveDualScaleUncertaintyLoss(cls_num_list=cls_num_list,
                                                        consistency_weight=0.1,  #  0.1-0.5
                                                        uncertainty_weight=1.0,  # 0.05-0.2
                                                        margin_weight=1.0,  # 0.05-0.1
                                                        margin=0.8  # 0.2
                                                        )
    else:
        raise ValueError("No such LossFunction: {}".format(config.loss_type))

    return criterion


class ClassBalancedFocalLoss(nn.Module):
    """
    parameter:
    - class_weights: Weight tensor for each class (higher weight for minority classes)
    - gamma: control the weight differences between easy and hard samples.（default: 2.0）
    - reduction: Loss reduction method, optional 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(self,
                 cls_num_list,
                 gamma=2.0,
                 reduction='mean',
                 beta=0.9999):
  
        super(ClassBalancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction  
        self.beta = beta
        self.class_weights = self.get_class_weights_for_loss(cls_num_list) 
        # self.class_weights = torch.tensor([1, 2], dtype=torch.float) 

    def get_class_weights_for_loss(self, cls_num_list):
        beta = 0.9999  
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(cls_num_list)
        return torch.tensor(weights, dtype=torch.float)

    def forward(self, logits, targets):
        """
        计算类平衡焦点损失

        参数:
        - logits: 模型预测（softmax之前的输出）
        - targets: 真实标签

        返回:
        - loss: 计算得到的类平衡焦点损失
        """

        self.class_weights = self.class_weights.to(logits.device)
        
        log_probs = F.log_softmax(logits, dim=1)

  
        batch_size = logits.size(0)
        log_pt = log_probs[torch.arange(batch_size), targets]
        pt = torch.exp(log_pt)  

        focal_weight = (1 - pt) ** self.gamma

        alpha_t = self.class_weights[targets]
        focal_weight = alpha_t * focal_weight

        loss = -log_pt * focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return super(ClassBalancedFocalLoss, self).to(device)
