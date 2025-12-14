import os
import os.path as osp
import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm
import pandas as pd
from tdc.utils import get_label_map
from tdc.multi_pred import Catalyst
import rdkit
from rdkit import Chem
from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Data

FILE_NAME = '.npy'


# class RxnCatDataSetOnTheFly(torch.utils.data.Dataset):
#     def __init__(self, root):
#         super().__init__()
#         self.root = root
#         self.raw_dir = osp.join(self.root, 'raw', 'uspto_catalyst.csv')
#         csv = Catalyst(name='USPTO_Catalyst', path=self.raw_dir).get_data()
#         label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=self.raw_dir)
#         # self.reactant = csv['X1'].tolist()
#         # self.product = csv['X2'].tolist()
#         # self.label = csv['Y'].tolist()
#         self.reactant = csv['Reactant'].tolist()
#         self.product = csv['Product'].tolist()
#         self.label = csv['Y'].tolist()
#         self.catlyst = [label_map[i] for i in self.label]
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         reactant = from_smiles(self.reactant[idx])
#         product = from_smiles(self.product[idx])
#         catlyst = from_smiles(self.catlyst[idx])
#         return reactant, product, catlyst, self.label[idx]


class RxnCatDataSet(Dataset):
    @property
    def raw_file_names(self):
        return ["uspto_catalyst.csv"]

    @property
    def processed_file_names(self):
        return [f"rxn_data_{idx}" + '.pt' for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.data_split))

    def __init__(self, root):
        self.root = root
        self.data_split = 'all'
        self.size_path = osp.join(self.root, "processed", "num_files" + '.pt')
        if osp.exists(self.size_path):
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        csv = Catalyst(name='USPTO_Catalyst', path=self.raw_dir).get_data()
        label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=self.raw_dir)
        print("label_map loaded...\n")
        self.reactant = csv['Reactant'].tolist()
        self.product = csv['Product'].tolist()
        self.label = csv['Y'].tolist()
        self.catlyst = [label_map[i] for i in self.label]
        super().__init__(root)
        print("super done...\n")

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            # cur_raw_dir = osp.join(self.raw_dir, raw_file_name)
            # csv = Catalyst(name='USPTO_Catalyst', path=cur_raw_dir).get_data()
            # label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=cur_raw_dir)
            csv = Catalyst(name='USPTO_Catalyst', path=self.raw_dir).get_data()
            label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=self.raw_dir)
            total, cur_id = len(csv), 0

            for idx in tqdm(range(total)):
                # reactant, product, label = csv.iloc[idx]['X1'], csv.iloc[idx]['X2'], csv.iloc[idx]['Y']
                # print(csv.columns)
                reactant, product, label = csv.iloc[idx]['Reactant'], csv.iloc[idx]['Product'], csv.iloc[idx]['Y']
                cur_catlyst = label_map[label]

                # Add data to columns
                self.reactant.append(reactant)
                self.product.append(product)
                self.catlyst.append(cur_catlyst)
                self.label.append(label)

                rxn_data = self._process_one(reactant, product, cur_catlyst)
                if rxn_data is not None:
                    torch.save(rxn_data, osp.join(self.processed_dir, f"rxn_data_{cur_id}" + '.pt'))
                    cur_id += 1
                else:
                    print(f"Error processing data: reactant：{reactant} product：{product} catlyst：{cur_catlyst}")
                    continue

            print(f"Completed the {raw_file_name} dataset to torch geometric format...")
            print(f"|total={cur_id}|\t|passed={total - cur_id}|", '*' * 90)
            self.size = cur_id
            # torch.save(self.size, self.size_path)
            torch.save(self.size, self.size_path)

    def _process_one(self, reactant, product, catlyst):
        try:
            r_graph = from_smiles(reactant)
            p_graph = from_smiles(product)
            c_graph = from_smiles(catlyst)
        except ValueError as e:
            # 处理异常情况，例如打印错误消息
            print(f"Error processing data: {e}")
            # 继续处理下一条数据
            return None

        return r_graph, p_graph, c_graph

    def len(self):
        return self.size

    def get(self, idx):
        rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}" + '.pt'))
        return rxn_data

    # def __getitem__(self, idx):
    #     reactant = from_smiles(self.reactant[idx])
    #     product = from_smiles(self.product[idx])
    #     catlyst = from_smiles(self.catlyst[idx])
    #     return reactant, product, catlyst, self.label[idx]
    def __getitem__(self, idx):
        try:
            reactant = from_smiles(self.reactant[idx])
            product = from_smiles(self.product[idx])
            catlyst = from_smiles(self.catlyst[idx])
            return reactant, product, catlyst, self.label[idx]
        except ValueError as e:
            # 记录错误信息和有问题的数据
            print(f"Error occurred at index {idx}: {e}")
            print(f"Problematic reactant data: {self.reactant[idx]}")
            print(f"Problematic product data: {self.product[idx]}")
            print(f"Problematic catlyst data: {self.catlyst[idx]}")
            # 可选：将错误信息和数据写入日志文件
            # with open('error_log.txt', 'a') as f:
            #     f.write(f"Error occurred at index {idx}: {e}\n")
            #     f.write(f"Problematic reactant data: {self.reactant[idx]}\n")
            #     f.write(f"Problematic product data: {self.product[idx]}\n")
            #     f.write(f"Problematic catlyst data: {self.catlyst[idx]}\n")
            # 重新抛出异常或处理异常
            raise


# -------- pyg-dataset 改为 pytorch-dataset ---------
# ----------------替换为lmdb读写数据-------------------
import lmdb
import pickle
from torch.utils.data import Dataset as torch_Dataset


class RxnCatDataSetwithLMDB(torch_Dataset):
    def __init__(self, root):
        self.root = root
        self.data_split = 'all_lmdb'
        self.prepare_lmdb_database()

    @property
    def raw_file_names(self):
        return ["uspto_catalyst.csv"]

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.data_split))

    @property
    def lmdb_file_path(self):
        return osp.join(self.processed_dir, 'data.lmdb')

    def open_lmdb_env(self):
        # Ensure a read-only connection for safety
        self.env = lmdb.open(self.lmdb_file_path, readonly=True, lock=False, readahead=False, meminit=False)

    def prepare_lmdb_database(self):
        if not osp.exists(self.lmdb_file_path):
            os.makedirs(osp.dirname(self.lmdb_file_path), exist_ok=True)

            csv = Catalyst(name='USPTO_Catalyst', path=self.raw_dir).get_data()
            label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=self.raw_dir)

            env = lmdb.open(self.lmdb_file_path, map_size=int(1e12))
            txn = env.begin(write=True)
            n_total = 0
            for idx, row in tqdm(csv.iterrows(), total=len(csv)):
                rxn_data = self.process_one(row, label_map)
                if rxn_data is not None:
                    txn.put(f"{n_total}".encode(), pickle.dumps(rxn_data))
                    n_total += 1
            txn.put("__len__".encode(), pickle.dumps(n_total))
            txn.put("__completed_key__".encode(), pickle.dumps(True))
            txn.commit()
            env.close()
        self.open_lmdb_env()

    def __len__(self):
        # 以只读方式打开LMDB环境
        assert self.env is not None
        with self.env.begin(write=False) as txn:
            # 获取样本数
            byteflow = txn.get("__len__".encode())
        if byteflow is not None:
            length = pickle.loads(byteflow)
            return length
        else:
            # 如果在数据库中没有找到样本数，返回0或抛出异常
            return 0

    def process_one(self, row, lable_map):
        reactant, product, label = row['Reactant'], row['Product'], row['Y']
        catalyst = lable_map[label]
        try:
            r_graph = from_smiles(reactant)
            p_graph = from_smiles(product)
            c_graph = from_smiles(catalyst)
        except ValueError as e:
            # 处理异常情况，例如打印错误消息
            print(f"Error processing data: {e}")
            # 继续处理下一条数据
            return None

        return r_graph, p_graph, c_graph, label

    def __getitem__(self, idx):
        assert self.env is not None
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(f"{idx}".encode())
        r_graph, p_graph, c_graph, label = pickle.loads(byteflow)
        return r_graph, p_graph, c_graph, label
