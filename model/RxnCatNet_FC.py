import sys
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model.CELosses import CEContrastiveLoss
from model.MolGNN import MolGNNLayers
import pickle

sys.path.append("..")
sys.path.append("./model")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool


class RxnCatNet(pl.LightningModule):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=12,
                 dropout=0.2,
                 num_classes=237,  # 假设分类类别数为462,现为237
                 num_gru_layer=4,
                 mol_encoder='gnn_gat_v1',
                 peak_lr=2e-4,
                 weight_decay=1e-5,
                 **kwargs):
        super().__init__()
        self.num_gru_layer = num_gru_layer
        self.base_gnn = MolGNNLayers(d_model=d_model, heads=nhead, num_layers=num_layers, dropout=dropout,
                                     mol_encoder=mol_encoder)
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_model = d_model
        # self.return_embedding = return_embedding
        self.peak_lr = peak_lr
        self.weight_decay = weight_decay
        self.mol_encoder = mol_encoder
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_gru_layer,
                          batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, num_classes)
        )

        # 新增分类指标
        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro"
        }

        # 验证指标
        self.val_acc = Accuracy(**metric_args)
        self.val_precision = Precision(**metric_args)
        self.val_recall = Recall(**metric_args)
        self.val_f1 = F1Score(**metric_args)
        self.val_auroc = AUROC(**metric_args)
        self.val_auprc = AveragePrecision(**metric_args)

        # 测试指标
        self.test_acc = Accuracy(**metric_args)
        self.test_precision = Precision(**metric_args)
        self.test_recall = Recall(**metric_args)
        self.test_f1 = F1Score(**metric_args)
        self.test_auroc = AUROC(**metric_args)
        self.test_auprc = AveragePrecision(**metric_args)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, mol_r, mol_p):
        mol_r = self.base_gnn(mol_r)
        mol_p = self.base_gnn(mol_p)
        mol_r = mol_r.unsqueeze(1)  # (batch, 1, d_model)
        mol_p = mol_p.unsqueeze(1)  # (batch, 1, d_model)
        sequence = torch.cat([mol_r, mol_p], dim=1)  # (batch, 2, d_model)
        mol_rea, _ = self.gru(sequence)  # 正确三维输入
        mol_rea = mol_rea[:, -1, :]  # (batch, d_model)

        # 分类logits
        logits = self.classifier(mol_rea)  # (batch, num_classes)
        return logits

    def _update_metrics(self, logits, labels, phase='val'):
        assert not torch.isnan(logits).any(), "Logits contain NaN!"
        assert not torch.isinf(logits).any(), "Logits contain Inf!"

        # 生成有效预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        if phase == 'val':
            self.val_acc(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)
            self.val_f1(preds, labels)
            self.val_auroc(probs, labels)
            self.val_auprc(probs, labels)
        elif phase == 'test':
            self.test_acc(preds, labels)
            self.test_precision(preds, labels)
            self.test_recall(preds, labels)
            self.test_f1(preds, labels)
            self.test_auroc(probs, labels)
            self.test_auprc(probs, labels)

    def compute_topk_accuracy(self, logits, labels, k_list=None):
        """返回固定顺序的元组 (top1, top3, top5, top10)"""
        if k_list is None:
            k_list = [1, 3, 5, 10]
        _, preds = torch.topk(logits, max(k_list), dim=1)
        correct = preds.eq(labels.view(-1, 1))

        acc_list = []
        for k in k_list:  # 保持顺序 [1,3,5,10]
            acc = correct[:, :k].any(dim=1).float().mean().item()
            acc_list.append(acc)

        return tuple(acc_list)

    def training_step(self, batch, batch_idx):
        mol_r, mol_p, mol_c, labels = batch
        labels = labels - 1  # 转换为0-based



        logits = self.forward(mol_r, mol_p)
        loss = self.criterion(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mol_r, mol_p, mol_c, labels = batch
        labels = labels - 1  # 转换为0-based
        logits = self.forward(mol_r, mol_p)
        loss = self.criterion(logits, labels)

        # 计算Top-k准确率（直接返回元组）
        top1_acc, top3_acc, top5_acc, top10_acc = self.compute_topk_accuracy(logits, labels, [1, 3, 5, 10])

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_top1_acc', top1_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_top3_acc', top3_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_top5_acc', top5_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_top10_acc', top10_acc, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss, "val_top1_acc": top1_acc, "val_top3_acc": top3_acc,
                "val_top5_acc": top5_acc, "val_top10_acc": top10_acc}

    def validation_epoch_end(self, outputs):
        avg_performance = self._avg_dicts(outputs)
        self._log_dict(avg_performance)

    def test_step(self, batch, batch_idx):
        mol_r, mol_p, mol_c, labels = batch
        labels = labels - 1  # 转换为0-based

        assert (labels >= 0).all() and (labels < self.hparams.num_classes).all(), \
            f"Invalid labels: {labels.min()}-{labels.max()}"


        logits = self.forward(mol_r, mol_p)
        loss = self.criterion(logits, labels)
        # 更新指标
        self._update_metrics(logits, labels, 'test')

        # 计算Top-k
        top1_acc, top3_acc, top5_acc, top10_acc = self.compute_topk_accuracy(logits, labels)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top1_acc', top1_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top3_acc', top3_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top5_acc', top5_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top10_acc', top10_acc, on_epoch=True, prog_bar=True, logger=True)

        return {
            "test_loss": loss,
            "top1": top1_acc,
            "top3": top3_acc,
            "top5": top5_acc,
            "top10": top10_acc
        }

    def test_epoch_end(self, outputs):
        # 过滤空批次
        outputs = [x for x in outputs if x is not None]
        if not outputs:
            raise ValueError("No test data available")

        # 记录所有指标
        self.log_dict({
            "test_acc": self.test_acc.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
            "test_f1": self.test_f1.compute(),
            "test_auroc": self.test_auroc.compute(),
            "test_auprc": self.test_auprc.compute(),
            "test_top1_loss": torch.tensor(np.mean([x["top1"] for x in outputs])),
            "test_top3_loss": torch.tensor(np.mean([x["top3"] for x in outputs])),
            "test_top5_loss": torch.tensor(np.mean([x["top5"] for x in outputs])),
            "test_top10_loss": torch.tensor(np.mean([x["top10"] for x in outputs]))
        }, prog_bar=True)

        # 重置指标
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()


    @staticmethod
    def _avg_dicts(colls):
        complete_dict = {key: [] for key, val in colls[0].items() if key != 'batch_size'}
        if "batch_size" in colls[0].keys():
            batch_sizes = [coll['batch_size'] for coll in colls]
        else:
            batch_sizes = [1] * len(colls)

        for col_index, coll in enumerate(colls):
            for key in complete_dict.keys():
                complete_dict[key].append(coll[key])
                complete_dict[key][col_index] *= batch_sizes[col_index]

        avg_dict = {key: sum(l) / sum(batch_sizes) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.peak_lr,
            weight_decay=self.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--seed', type=int, default=43)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--num_layers', type=int, default=6)
        parser.add_argument('--num_encoder_layer', type=int, default=2)
        parser.add_argument('--dim_feedforward', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.4)
        parser.add_argument('--norm_first', default=False, action='store_true')
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument("--warmup_updates", type=float, default=2000)
        parser.add_argument("--tot_updates", type=float, default=1e10)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        return parser

