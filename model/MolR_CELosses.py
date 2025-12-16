from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from pytorch_lightning.utilities.seed import seed_everything


# 侧重于催化剂的语义相似性或化学性质的相似性，推荐这个
class CEContrastiveLoss(nn.Module):
    def __init__(self,):
        super(CEContrastiveLoss, self).__init__()

    def forward(self, feature_map_1, feature_map_2, labels, margin):
        """
        :param margin:
        :param feature_map_2: [batch_size, d_model]
        :param feature_map_1:[batch_size, d_model]
        :param labels:[batch_size]
        :return:
        """
        # feature_map_1 = F.normalize(feature_map_1, dim=-1)
        # feature_map_2 = F.normalize(feature_map_2, dim=-1)
        # dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
        dist = torch.cdist(feature_map_1, feature_map_2, p=2)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).to(dist)
        pos = dist * mask
        pos = pos[pos != 0]
        # if torch.cuda.is_available():
        #     mask = mask.cuda()
        # neg = (1 - mask) * dist + mask * args.margin
        # neg = torch.relu(args.margin - neg)
        neg = (1 - mask) * dist + mask * margin
        neg = torch.relu(margin - neg)
        # loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)
        loss = torch.mean(pos) + torch.sum(neg) / (labels.size(0) * labels.size(0) - pos.size(0))
        return loss

