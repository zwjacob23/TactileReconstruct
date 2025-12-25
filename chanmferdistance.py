import torch
import torch.nn as nn
import numpy as np


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, preds, targets):
        # preds and targets shape: (batch_size, num_points, 3)
        batch_size, num_points, _ = preds.size()

        preds = preds.unsqueeze(1).repeat(1, num_points, 1, 1)
        targets = targets.unsqueeze(2).repeat(1, 1, num_points, 1)

        dist = torch.norm(preds - targets, dim=3)
        min_dist_pred_to_target, _ = torch.min(dist, dim=2)
        min_dist_target_to_pred, _ = torch.min(dist, dim=1)

        chamfer_distance = torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)
        return chamfer_distance