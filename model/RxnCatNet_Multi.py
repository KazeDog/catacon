import sys
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model.CELosses_InfoNCE_Contrastive import CEContrastiveLoss
# from model.CELosses_InfoNCE_Contrastive import CEContrastiveLoss
from model.MolGNN import MolGNNLayers
import pickle
from transformers import T5Tokenizer, T5EncoderModel


sys.path.append("..")
sys.path.append("./model")


class RxnCatNet(pl.LightningModule):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=12,
                 dropout=0.2,
                 warmup_updates=6e4,
                 tot_updates=1e6,
                 peak_lr=2e-4,
                 end_lr=1e-9,
                 weight_decay=0.99,
                 # return_embedding=False,
                 num_gru_layer=4,
                 temperature=0.5,
                 mol_encoder='gnn_gat_v1',
                 alpha=0.01):

        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        # self.return_embedding = return_embedding
        self.num_gru_layer = num_gru_layer
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.warmup_updates = warmup_updates
        self.peak_lr = peak_lr
        self.tot_updates = tot_updates
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.mol_encoder = mol_encoder
        self.base_gnn = MolGNNLayers(d_model=d_model, heads=nhead, num_layers=num_layers, dropout=dropout,
                                     mol_encoder=mol_encoder)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_gru_layer,
                          batch_first=True, bidirectional=False)
        self.criterion = CEContrastiveLoss(temperature=self.temperature)
        self.save_hyperparameters()

        metric_args = {
            "task": "multiclass",
            "num_classes": 237,
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

        with open('data/class_batch.pkl', 'rb') as f:
            self.mol_class = pickle.load(f)
        self.alpha = alpha

        

    def forward(self, mol_r, mol_p, mol_c, mol_class, label=None, log=False):
        mol_r = self.base_gnn(mol_r)
        mol_p = self.base_gnn(mol_p)
        mol_c = self.base_gnn(mol_c)
        # mol_rea, _ = self.gru(torch.cat([mol_r, mol_p], dim=1))
        mol_r = mol_r.unsqueeze(1)  # (batch, 1, d_model)
        mol_p = mol_p.unsqueeze(1)  # (batch, 1, d_model)
        sequence = torch.cat([mol_r, mol_p], dim=1)  # (batch, 2, d_model)
        mol_rea, _ = self.gru(sequence)  # 正确三维输入
        mol_rea = mol_rea[:, -1, :]  # (batch, d_model)
        mol_class = self.base_gnn(mol_class)
        return mol_rea, mol_c, mol_class

    def get_contrastive_loss(self, mol_rea, mol_c, label):
        ce_loss = self.criterion(mol_rea, mol_c, label, alpha=self.alpha)
        return ce_loss

    def _get_class_probs(self, query_feature, class_features):
        """新增方法：获取类别概率分布"""
        # 计算相似度得分
        query_feature = F.normalize(query_feature, p=2, dim=1)
        class_features = F.normalize(class_features, p=2, dim=1)
        logits = torch.mm(query_feature, class_features.t()) / self.temperature
        
        # 转换为概率分布
        probs = F.softmax(logits, dim=1)
        return probs

    def _update_metrics(self, probs, labels, phase='val'):
        """修正指标更新方法"""
        preds = torch.argmax(probs, dim=1)

        # 明确调用update方法
        if phase == 'val':
            self.val_acc.update(preds, labels)
            self.val_precision.update(preds, labels)
            self.val_recall.update(preds, labels)
            self.val_f1.update(preds, labels)
            self.val_auroc.update(probs, labels)  # AUROC需要概率值
            self.val_auprc.update(probs, labels)  # AUPRC需要概率值
        elif phase == 'test':
            self.test_acc.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_recall.update(preds, labels)
            self.test_f1.update(preds, labels)
            self.test_auroc.update(probs, labels)
            self.test_auprc.update(probs, labels)

    def get_topk(self, feature_map_1, feature_map_2, label, k_list):
        # 添加维度校验
        assert feature_map_1.size(1) == feature_map_2.size(1), "Feature dimension mismatch"

        # 计算logits
        feature_map_1 = F.normalize(feature_map_1, dim=-1)
        feature_map_2 = F.normalize(feature_map_2, dim=-1)
        logits = torch.mm(feature_map_1, feature_map_2.t()) / self.temperature

        # 数值稳定处理
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = torch.softmax(logits, dim=1)

        acc_list = []
        for k in k_list:
            # 使用topk优化
            _, topk_indices = torch.topk(probs, k, dim=1, sorted=True)
            correct_topk = torch.any(topk_indices == (label.unsqueeze(1) - 1), dim=1)
            topk_accuracy = correct_topk.float().mean().item()  # 合并处理逻辑
            acc_list.append(topk_accuracy)
        return tuple(acc_list)

    def training_step(self, batch, batch_idx):
        mol_r, mol_p, mol_c, label = batch
        mol_class = self.mol_class.to('cuda')
        mol_rea_fea, mol_c_fea, mol_class_fea = self.forward(mol_r, mol_p, mol_c, mol_class)
        loss = self.get_contrastive_loss(mol_rea_fea, mol_c_fea, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # top1_acc, top3_acc, top5_acc, top10_acc = self.get_topk(mol_rea_fea, mol_class_fea, label, [1, 3, 5, 10])
        # self.log('train_top1_acc', top1_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_top3_acc', top3_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_top5_acc', top5_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_top10_acc', top10_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mol_r, mol_p, mol_c, label = batch
        mol_class = self.mol_class.to('cuda')
        mol_rea_fea, mol_c_fea, mol_class_fea = self.forward(mol_r, mol_p, mol_c, mol_class)
        loss = self.get_contrastive_loss(mol_rea_fea, mol_c_fea, label)
        top1_acc, top3_acc, top5_acc, top10_acc = self.get_topk(mol_rea_fea, mol_class_fea, label, [1, 3, 5, 10])

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
        mol_r, mol_p, mol_c, label = batch
        mol_class = self.mol_class.to('cuda')
        mol_rea_fea, mol_c_fea, mol_class_fea = self.forward(mol_r, mol_p, mol_c, mol_class)

        # 确保至少有1个样本
        if mol_rea_fea.shape[0] == 0:
            return None  # 跳过空批次

        loss = self.get_contrastive_loss(mol_rea_fea, mol_c_fea, label)
        top1_acc, top3_acc, top5_acc, top10_acc = self.get_topk(mol_rea_fea, mol_class_fea, label, [1, 3, 5, 10])

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top1_acc', top1_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top3_acc', top3_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top5_acc', top5_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_top10_acc', top10_acc, on_epoch=True, prog_bar=True, logger=True)

        # 获取类别概率
        probs = self._get_class_probs(mol_rea_fea, mol_class_fea)

        # 更新指标
        self._update_metrics(probs, label - 1, phase='test')

        # 计算top-k
        top1, top3, top5, top10 = self.get_topk(mol_rea_fea, mol_class_fea, label, [1, 3, 5, 10])

        # return {
        #     "test_loss": loss, "test_top1_acc": top1_acc, "test_top3_acc": top3_acc,
        #         "test_top5_acc": top5_acc, "test_top10_acc": top10_acc
        # }
        
        return {
            "test_loss": self.get_contrastive_loss(mol_rea_fea, mol_c_fea, label),
            "probs": probs,
            "labels": label-1,
            "test_top1": top1,
            "test_top3": top3,
            "test_top5": top5,
            "test_top10": top10
        }

    def test_epoch_end(self, outputs):
        # 过滤空批次
        outputs = [x for x in outputs if x is not None]
        if not outputs:
            raise ValueError("No test batches available")

        # 安全计算指标
        self.log_dict({
            "test_acc": self.test_acc.compute() if self.test_acc._update_called else torch.tensor(0.0),
            "test_precision": self.test_precision.compute() if self.test_precision._update_called else torch.tensor(
                0.0),
            "test_recall": self.test_recall.compute() if self.test_recall._update_called else torch.tensor(0.0),
            "test_f1": self.test_f1.compute() if self.test_f1._update_called else torch.tensor(0.0),
            "test_auroc": self.test_auroc.compute() if self.test_auroc._update_called else torch.tensor(0.0),
            "test_auprc": self.test_auprc.compute() if self.test_auprc._update_called else torch.tensor(0.0),
            "test_top1_loss": torch.tensor(np.mean([x["test_top1"] for x in outputs])),
            "test_top3_loss": torch.tensor(np.mean([x["test_top3"] for x in outputs])),
            "test_top5_loss": torch.tensor(np.mean([x["test_top5"] for x in outputs])),
            "test_top10_loss": torch.tensor(np.mean([x["test_top10"] for x in outputs]))
        })

        # 强制重置指标
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.predict_state = True
        mol_r, mol_p, mol_c, label = batch
        mol_class = self.mol_class.to('cuda')
        # 获取特征表示
        mol_rea_fea, mol_c_fea, mol_class_fea = self.forward(mol_r, mol_p, mol_c, mol_class)

        # 返回需要的特征字典
        return {
            "mol_rea_fea": mol_rea_fea,  # 分子反应物特征
            "mol_c_fea": mol_c_fea,  # 分子催化剂特征
            "label": label  # 原始标签
        }

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

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        # )
        # lr_scheduler = {
        #     "scheduler": PolynomialDecayLR(
        #         optimizer,
        #         warmup_updates=self.warmup_updates,
        #         tot_updates=self.tot_updates,
        #         lr=self.peak_lr,
        #         end_lr=self.end_lr,
        #         power=1.0,
        #     ),
        #     "name": "learning_rate",
        #     "interval": "step",
        #     "frequency": 1,
        # }

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.peak_lr,
            weight_decay=self.weight_decay)
        return optimizer

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),  # 传入模型参数
    #         lr=self.peak_lr,
    #         weight_decay=self.weight_decay
    #     )
    #
    #     # 确保 warmup_updates 和 tot_updates 是整数
    #     warmup_updates = int(self.warmup_updates)
    #     tot_updates = int(self.tot_updates)
    #
    #     scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_updates)
    #     scheduler2 = CosineAnnealingLR(optimizer, T_max=tot_updates, eta_min=self.end_lr)
    #
    #     scheduler = SequentialLR(
    #         optimizer,
    #         schedulers=[scheduler1, scheduler2],
    #         milestones=[warmup_updates]
    #     )
    #
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "frequency": 1
    #         }
    #     }
