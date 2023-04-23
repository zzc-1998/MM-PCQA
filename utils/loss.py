import os, argparse, time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


class L2RankLoss(torch.nn.Module):
    """
    L2 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L2RankLoss, self).__init__()
        self.l2_w = 1
        self.rank_w = 1
        self.hard_thred = 1
        self.use_margin = False

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l2_loss = F.mse_loss(preds, gts) * self.l2_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l2_loss + rank_loss * self.rank_w
        return loss_total

