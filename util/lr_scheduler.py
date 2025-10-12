import torch
from torch import nn
import math
from torch.optim import lr_scheduler


def create_lr_scheduler(optimizer,
                        config,
                        num_step:int,
                        warmup=True,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    epochs = config.epoch
    warmup_epochs = config.warmup_epochs
    if config.lr_scheduler == 'step':
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0
        def f(x):
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                return warmup_factor * (1 - alpha) + alpha
            else:
                current_step = (x - warmup_epochs * num_step)
                cosine_steps = (epochs - warmup_epochs) * num_step
                return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    elif config.lr_scheduler == 'cos_lr_step':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * num_step, eta_min=config.end_lr_cos)
    elif config.lr_scheduler == 'cos_lr_epoch':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.end_lr_cos)
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler
