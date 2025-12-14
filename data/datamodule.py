import pytorch_lightning as pl
import torch
from torch_geometric import loader
from tdc.multi_pred import Catalyst
import os
# from data.datasets import RxnCatDataSetwithLMDB
from data.datasets import RxnCatDataSetwithLMDB
from torch.utils import data
from torch.utils.data.dataloader import default_collate


class MolDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root,
            batch_size,
            split_names=None,
            num_workers=None,
            # pin_memory=True,
            pin_memory=False,
            shuffle=True,
            seed=114,
            prompt=False,
            on_the_fly=False,
    ):
        super().__init__()
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.dataset_dict = {}
        self.prompt = prompt
        self.on_the_fly = on_the_fly

        if split_names is None:
            split_names = ['train', 'valid', 'test']

        self.split_names = split_names
        self.data_loader_fn = loader.DataLoader

        # raw_data = Catalyst(name='USPTO_Catalyst', path='data/uspto')
        cur_dataset = RxnCatDataSetwithLMDB(root)
        # 数据集随机分割，比例均为原数据集的
        train_size = int(len(cur_dataset) * 0.8)
        valid_size = int(len(cur_dataset) * 0.1)
        test_size = len(cur_dataset) - train_size - valid_size
        cur_split = data.random_split(cur_dataset, lengths=[train_size, valid_size, test_size],
                                      generator=torch.Generator().manual_seed(seed))
        for i, split_name in enumerate(split_names):
            self.dataset_dict[split_name] = cur_split[i]

    def train_dataloader(self):
        print('train_subset:', self.dataset_dict[self.split_names[0]].__len__())
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[0]],
            # self.dataset_dict[self.split_names[0]].dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            # collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[1]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            # collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[2]],
            # self.dataset_dict[self.split_names[0]].dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            # collate_fn=custom_collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


