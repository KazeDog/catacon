import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.utilities import seed
from model.RxnCatNet import RxnCatNet
from data.datamodule import MolDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import defaultdict
import importlib

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.autograd.set_detect_anomaly(True)


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = RxnCatNet.add_model_specific_args(parser)

    # trainer configuration
    parser.add_argument('--epochs', type=int, default=2000)
    # parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Gradient clipping value')
    # parser.add_argument('--acc_batches', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default='tb_logs')
    # parser.add_argument('--name', type=str, default='yield')
    parser.add_argument('--name', type=str, default='Y')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--predict', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--patience', type=int, default=20)

    # dataset configuration
    parser.add_argument('--dataset', type=str, default="data/uspto")
    # parser.add_argument('--not_fast_read', default=False, action='store_true')
    # parser.add_argument('--use_3d_info', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--dataset_format', type=str, default='buch-hart')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--mol_encoder', type=str, default='gnn_gat_v1')

    args = parser.parse_args()
    print(args)

    seed.seed_everything(args.seed)

    print("Building DataModule...")
    dm = build_datamodule(args)
    print("Finished DataModule.")

    print("Building Model...")
    model = build_model(args)
    print("Finished Model...")

    print("Building Trainer...")
    trainer = build_trainer(args)
    print("Finished Trainer...")

    if not args.test and not args.predict:
        trainer.fit(model, dm)
        # trainer.fit(model, dm, ckpt_path="model_1293.pth")
        print('Finished training..')
        print(args)

    elif args.test:
        print('Testing...')
        trainer.test(model, dm)
        print('Finished predicting...')

    elif args.predict:
        print('predict...')
        root = args.dataset
        os.makedirs(root, exist_ok=True)
        outputs = trainer.predict(model, dm)

        complete_dict = defaultdict(list)
        for coll in outputs:
            for key in coll.keys():
                complete_dict[key].append(coll[key])

        for key in complete_dict.keys():
            complete_dict[key] = torch.cat(complete_dict[key], dim=0)

        label_indices = defaultdict(list)
        for idx, label in enumerate(complete_dict["label"]):
            label_indices[label.item()].append(idx)

        print("每个类别的样本数:")
        for label, indices in label_indices.items():
            print(f'标签类别: {label} 样本数: {len(indices)}')

        all_labels = np.array(list(label_indices.keys()))
        selected_labels = np.random.choice(all_labels, 10, replace=False)

        combined_data = {"mol_rea_fea": [], "mol_c_fea": [], "label": []}
        for label in selected_labels:
            indices = label_indices[label]

            if len(indices) > 20:
                indices = np.random.choice(indices, 20, replace=False)
            else:

                print(f'标签类别: {label} 当前特征数: {len(indices)}')
            for idx in indices:
                combined_data["mol_rea_fea"].append(complete_dict["mol_rea_fea"][idx].unsqueeze(0))
                combined_data["mol_c_fea"].append(complete_dict["mol_c_fea"][idx].unsqueeze(0))
                combined_data["label"].append(complete_dict["label"][idx].unsqueeze(0))

        # 合并所有特征数据
        for key in combined_data.keys():
            combined_data[key] = torch.cat(combined_data[key], dim=0)

        torch.save(combined_data, os.path.join(root, 'tsne_0228.pt'))



def build_trainer(args):
    logger = TensorBoardLogger(args.log_dir, name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, mode='min', verbose=True,)
    # early_stop = EarlyStopping(monitor="valid_acc", patience=args.patience, mode='max')

    trainer = Trainer(
        # accelerator='cpu',
        accelerator='gpu',
        # strategy="auto",
        # strategy='ddp',
        logger=logger,
        gpus=list(map(int, args.cuda.split(','))),
        max_epochs=args.epochs,
        # accumulate_grad_batches=args.acc_batches,
        # callbacks=[lr_monitor, checkpoint_cb, early_stop],
        callbacks=[lr_monitor, checkpoint_cb],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        detect_anomaly=False,
        # precision=16 if not args.predict else 32,
        precision=32,
        # auto_lr_find='peak_lr',
    )
    return trainer


def build_datamodule(args):
    dm = MolDataModule(
        root=args.dataset,
        batch_size=args.batch_size,
        # fast_read=not args.not_fast_read,
        num_workers=args.num_workers,
        # predict=args.predict or args.test,
        # dataset_format=args.dataset_format,
        seed=args.seed,
    )
    return dm


def build_model(args):
    if args.model_path == '':
        model = RxnCatNet(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            # r_layer=args.r_layer,
            # fusion_layer=args.fusion_layer,
            # dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            # known_rxn_cnt=args.known_rxn_cnt,
            # norm_first=args.norm_first,
            # activation=args.activation,
            weight_decay=args.weight_decay,
            # use_3d_info=args.use_3d_info,
            # warmup_updates=args.warmup_updates,
            # tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            # end_lr=args.end_lr,
            # max_single_hop=args.max_single_hop,
            # use_dist_adj=not args.not_use_dist_adj,
            # use_contrastive=not args.not_use_contrastive,
            alpha=args.alpha,
            mol_encoder=args.mol_encoder,
        )
    else:
        model = RxnCatNet.load_from_checkpoint(
            args.model_path,
            # strict=True,
            strict=False,
        )

    return model


if __name__ == '__main__':
    main()


