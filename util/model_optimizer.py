import torch.optim as optim
import torch
from scipy.interpolate import interp1d
import numpy as np



def create_optimizer(config, model_parameter):
    """
    :param config:
    :param model_parameter:
    :return:
    """
    if config.model_optimizer == 'Adam':
        optimizer = optim.Adam(model_parameter,
                               lr=config.base_learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=1e-4)
    elif config.model_optimizer == 'SGD':
        optimizer = optim.SGD(model_parameter,
                              lr=config.base_learning_rate,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=True)
    else:
        raise ValueError("No such optimizer: {}".format(config.model_optimizer))

    return optimizer