from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from pytorch_lightning.utilities.seed import seed_everything
import math


# 当前是负抽样修正版本

class CEContrastiveLoss(nn.Module):
    def __init__(self, temperature, c=0.1):
        super(CEContrastiveLoss, self).__init__()
        self.temperature = temperature
        # self.alpha = alpha
        self.c = c

    def forward(self, feature_map_1, feature_map_2, labels, alpha=0.01):
        """
        :param alpha:
        :param feature_map_1: [batch_size, d_model]
        :param feature_map_2: [batch_size, d_model]
        :param labels: [batch_size]
        :return: loss
        """
        # 计算余弦相似度
        feature_map_1 = F.normalize(feature_map_1, dim=-1)
        feature_map_2 = F.normalize(feature_map_2, dim=-1)
        # cosine_similarity = F.cosine_similarity(feature_map_1.unsqueeze(1), feature_map_2.unsqueeze(0),
        #                                         dim=2) / self.temperature

        logits = (torch.matmul(feature_map_1, feature_map_2.t().contiguous())) / self.temperature

        # 生成掩码
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # 计算正样本和负样本的对比损失
        logits = torch.exp(logits)
        positive_sim = logits * mask
        positive_mean = positive_sim.sum(dim=1) / mask.sum(dim=1)
        negative_sim = logits * (1 - mask)
        negative_mean = negative_sim.sum(dim=1) / (1 - mask).sum(dim=1)

        # pos_count = mask.sum(dim=1)
        # if pos_count == 0:
        #     positive_mean = torch.zeros_like(pos_count, dtype=torch.float32)
        # else:
        #     positive_mean = (logits * pos_count) / pos_count
        # neg_count = (1 - mask).sum(dim=1)
        # if neg_count == 0:
        #     negative_mean = torch.zeros_like(neg_count, dtype=torch.float32)
        # else:
        #     negative_mean = (logits * neg_count) / neg_count

        # positive_sim = cosine_similarity * mask
        # negative_sim = cosine_similarity * (1 - mask)


        # 计算 \mu_x
        N = feature_map_1.shape[0] - 2
        # mu_x_raw = ((1 - alpha * self.c) / (1 - alpha)) * negative_sim.mean(dim=1) - (
        #         alpha * (1 - self.c) / (1 - alpha)) * positive_sim.mean(dim=1)
        mu_x_raw = ((1 - alpha * self.c) / (1 - alpha)) * negative_mean - (
                alpha * (1 - self.c) / (1 - alpha)) * positive_mean
        mu_x = torch.max(mu_x_raw, torch.tensor(math.exp(-1), device=mu_x_raw.device))

        # loss = (- torch.log(positive_sim / positive_sim + N * mu_x).sum(dim=1)) / mask.sum(dim=1)
        loss_raw = - torch.log(logits / (logits + N * mu_x))
        positive_loss = loss_raw * mask

        # 总的对比学习损失
        loss = positive_loss[positive_loss != 0].mean()

        return loss
