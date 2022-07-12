import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def log_cosh_loss(y_pred, y_true):
    def _log_cosh(x):
        return x + F.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_ture):
        return log_cosh_loss(y_pred, y_ture)
