from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from pytorch_lightning.utilities.seed import seed_everything


# 侧重于催化剂的语义相似性或化学性质的相似性，推荐这个
class CEContrastiveLoss(nn.Module):
    def __init__(self, temperature=1., p_1=0, alpha=1, eps=1e-20):
        super(CEContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha

    def forward(self, feature_map_1, feature_map_2, labels):
        """
        :param feature_map_2: [batch_size, d_model]
        :param feature_map_1:[batch_size, d_model]
        :param labels:[batch_size]
        :return:
        """
        # # sim-1 is for numerical stability
        # # logits = (self.cal_similarity(torch.mm(feature_map_1, feature_map_2.t().contiguous())) - 1) / self.temperature
        # assert not torch.isnan(feature_map_1).any(), "29:NaN detected in feature_map_1"
        # assert not torch.isnan(feature_map_2).any(), "30:NaN detected in feature_map_2"
        # assert not torch.isnan(labels).any(), "31:NaN detected in labels"
        # 设置新的随机种子
        # new_seed = np.random.randint(0, 10000)
        # np.random.seed(new_seed)
        # torch.manual_seed(new_seed)

        feature_map_1 = F.normalize(feature_map_1, dim=-1)
        feature_map_2 = F.normalize(feature_map_2, dim=-1)

        feature_map_all = torch.cat((feature_map_1, feature_map_2), dim=0)

        # logits = (torch.matmul(feature_map_1, feature_map_2.t().contiguous()) - 1) / self.temperature
        logits = (torch.matmul(feature_map_all, feature_map_all.t().contiguous())) / self.temperature
        # assert not torch.isnan(logits).any(), "33:NaN detected in logits"

        # labels = labels.unsqueeze(1)
        # labels_c = torch.arange(1, 463).unsqueeze(0).to(labels.device, dtype=labels.dtype)
        # mask = torch.eq(labels, labels_c).to(logits)

        labels = torch.cat((labels, labels), dim=0)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).to(logits)

        # random_columns = np.random.permutation(logits.shape[1])
        # logits = logits[:, random_columns]
        # mask = mask[:, random_columns]

        # mask = torch.where(mask == 0, -1., 1.)
        # assert not torch.isnan(mask).any(), "40:NaN detected in mask"
        diag_mask = torch.ones_like(mask).fill_diagonal_(0)
        # assert not torch.isnan(logits_mask).any(), "42:NaN detected in logits_mask"
        mask = mask * diag_mask
        # assert not torch.isnan(mask).any(), "44:NaN detected in mask"
        # exp_logits = logits.exp() * logits_mask
        # try:
        #     assert not torch.isnan(exp_logits).any(), "39:NaN detected in exp_logits"
        # except AssertionError as e:
        #     print("Assertion Error:", e)
        #     print("logits values:", logits)
        #     raise  # 重新抛出异常以确保程序可以正常响应断言错误

        # log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdims=True)
        exp_logits = torch.exp(logits) * diag_mask
        # exp_logits = torch.exp(logits)
        # log_prob = logits * mask - torch.log((logits * diag_mask).sum(dim=1, keepdim=True) + self.eps)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        # log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        # log_prob = logits_p - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        # mean_log_prob = (log_prob * mask).sum(dim=1)                                                 # V1
        mean_log_prob = ((log_prob * mask).sum(dim=1)) / mask.sum(dim=1).clamp(min=self.eps)       # V2
        # loss = (- mean_log_prob * self.alpha).mean(dim=0)
        # mean_log_prob = ((log_prob * mask).sum(dim=1)) / (mask.sum(dim=1) + self.eps)
        # mean_log_prob = ((log_prob * mask).sum(dim=1) + self.eps) / (mask.sum(dim=1) + self.eps) # 原來的
        # assert not torch.isnan(mean_log_prob).any(), "43:NaN detected in mean_log_prob"

        # loss = (- mean_log_prob[mean_log_prob != 0] * self.alpha).mean(dim=0)
        # 原來的
        loss = (- mean_log_prob * self.alpha).mean(dim=0)                                             # V1
        # loss = (- mean_log_prob * self.alpha).sum(dim=0)                                            # V2
        # assert not torch.isnan(loss).any(), "46:NaN detected in loss"
        return loss

    # def cal_similarity_for_matrix(self, feature_maps):
    #     """
    #     :param feature_maps: [batch_size, seq_len, d_model]
    #     :return: matrix similarity
    #     """
    #     word_vectors = (F.normalize(feature_maps, dim=2)).transpose(0, 1)
    #     # word_vectors = data0.transpose(0, 1)
    #     word_smi = torch.bmm(word_vectors, word_vectors.transpose(1, 2))
    #     word_smi = word_smi.mean(dim=0).squeeze()
    #
    #     semantic_vectors = (F.normalize(feature_maps, dim=1)).transpose(0, 2)
    #     # semantic_vectors = data0.transpose(0, 2)
    #     semantic_smi = torch.bmm(semantic_vectors.transpose(1, 2), semantic_vectors)
    #     semantic_smi = semantic_smi.mean(dim=0).squeeze()
    #     return self.p_1 * word_smi + (1 - self.p_1) * semantic_smi
    #
    # def cal_similarity(self, feature_maps):
    #     """
    #     :param feature_maps: [batch_size, d_model]
    #     :return: vector similarity
    #     """
    #     word_vectors = (F.normalize(feature_maps, dim=2)).transpose(0, 1)
    #     # word_vectors = data0.transpose(0, 1)
    #     word_smi = torch.bmm(word_vectors, word_vectors.transpose(1, 2))
    #     word_smi = word_smi.mean(dim=0).squeeze()
    #
    #     semantic_vectors = (F.normalize(feature_maps, dim=1)).transpose(0, 2)
    #     # semantic_vectors = data0.transpose(0, 2)
    #     semantic_smi = torch.bmm(semantic_vectors.transpose(1, 2), semantic_vectors)
    #     semantic_smi = semantic_smi.mean(dim=0).squeeze()
    #     return self.p_1 * word_smi + (1 - self.p_1) * semantic_smi
