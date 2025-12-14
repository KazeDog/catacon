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
import lmdb
import pickle
from torch.utils.data import Dataset as torch_Dataset
import torch
from torch_geometric.data import Data, Batch


def process_one(labels, label_map):
    catalyst = label_map[labels]
    try:
        c_graph = from_smiles(catalyst)
    except ValueError as e:
        print(f"Error processing data: {e}")
        return None

    return c_graph, labels


def concatenate_data(data_list):
    concatenated_r_graphs = []
    concatenated_labels = []

    for rxn_data in data_list:
        r_graph, label = rxn_data
        concatenated_r_graphs.append(r_graph)
        concatenated_labels.append(label)

    concatenated_data_batch = {
        'x': concatenated_r_graphs,
        'label': concatenated_labels
    }

    return concatenated_data_batch


def save_data_batch(data_batch, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_batch, f)


def process_data(data_batch):
    data_list = []
    for x, label in zip(data_batch['x'], data_batch['label']):
        data_list.append(Data(x=x.x, edge_index=x.edge_index, edge_attr=x.edge_attr, smiles=x.smiles))
    batch = Batch.from_data_list(data_list)
    # batch['label'] = torch.tensor(data_batch['label'], dtype=torch.long)
    return batch


def save_batch(batch, filename):
    with open(filename, 'wb') as f:
        pickle.dump(batch, f)


label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path='uspto/raw')
data_list = []

# for label in tqdm(range(1, 463)):
for label in tqdm(range(1, 238)):
    rxn_data = process_one(label, label_map)
    if label == 1:
        print(rxn_data)
    if rxn_data is not None:
        data_list.append(rxn_data)

# 拼接数据并保存
data_batch = concatenate_data(data_list)
processed_data = process_data(data_batch)
# print(processed_data)
save_batch(processed_data, 'class_batch.pkl')

# def process_one(labels, lable_map):
#     catalyst = lable_map[labels]
#     try:
#         # r_graph = from_smiles(reactant)
#         # p_graph = from_smiles(product)
#         c_graph = from_smiles(catalyst)
#     except ValueError as e:
#         # 处理异常情况，例如打印错误消息
#         print(f"Error processing data: {e}")
#         # 继续处理下一条数据
#         return None
#
#     return c_graph, labels
#
#
# def concatenate_data(data_list):
#     concatenated_r_graphs = []
#     concatenated_labels = []
#
#     for rxn_data in data_list:
#         r_graph, label = rxn_data
#         concatenated_r_graphs.append(r_graph)
#         concatenated_labels.append(label)
#
#     concatenated_data_batch = {
#         'x': concatenated_r_graphs,
#         'label': concatenated_labels
#     }
#
#     return concatenated_data_batch
#
#
# def save_data_batch(data_batch, filename):
#     with open(filename, 'wb') as f:
#         pickle.dump(data_batch, f)
#
#
# # 拼接数据并保存
# def process_data(data_batch):
#     data_list = []
#     for x, label in zip(data_batch['x'], data_batch['label']):
#         data_list.append(Data(x=x.x, edge_index=x.edge_index, edge_attr=x.edge_attr, smiles=x.smiles))
#     batch = Batch.from_data_list(data_list)
#     batch['label'] = torch.tensor(data_batch['label'], dtype=torch.long)
#     return batch
#
#
# label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path='uspto/raw')
# data_list = []
#
# for label in tqdm(range(1, 889)):
#     rxn_data = process_one(label, label_map)
#     if label == 1:
#         print(rxn_data)
#     if rxn_data is not None:
#         data_list.append(rxn_data)
# # 拼接数据并保存
# data_batch = concatenate_data(data_list)
# processed_data = process_data(data_batch)
# print(processed_data)
# save_data_batch(data_batch, 'data_batch.pkl')

# class RxnCatDataSetwithLMDB(torch_Dataset):
#     def __init__(self, root):
#         self.root = root
#         self.data_split = 'all_lmdb_class'
#         self.prepare_lmdb_database()
#
#     @property
#     def raw_file_names(self):
#         return ["uspto_catalyst.csv"]
#
#     @property
#     def raw_dir(self):
#         return osp.join(self.root, 'raw')
#
#     @property
#     def processed_dir(self) -> str:
#         return osp.join(self.root, osp.join("processed", self.data_split))
#
#     @property
#     def lmdb_file_path(self):
#         return osp.join(self.processed_dir, 'data.lmdb')
#
#     def open_lmdb_env(self):
#         # Ensure a read-only connection for safety
#         self.env = lmdb.open(self.lmdb_file_path, readonly=True, lock=False, readahead=False, meminit=False)
#
#     def prepare_lmdb_database(self):
#         if not osp.exists(self.lmdb_file_path):
#             os.makedirs(osp.dirname(self.lmdb_file_path), exist_ok=True)
#
#             csv = Catalyst(name='USPTO_Catalyst', path=self.raw_dir).get_data()
#             label_map = get_label_map(name='USPTO_Catalyst', task='Catalyst', path=self.raw_dir)
#
#             env = lmdb.open(self.lmdb_file_path, map_size=int(1e12))
#             txn = env.begin(write=True)
#             n_total = 0
#             for label in tqdm(range(1, 889)):
#                 rxn_data = self.process_one(label, label_map)
#                 if rxn_data is not None:
#                     txn.put(f"{n_total}".encode(), pickle.dumps(rxn_data))
#                     n_total += 1
#
#             txn.put("__len__".encode(), pickle.dumps(n_total))
#             txn.put("__completed_key__".encode(), pickle.dumps(True))
#             txn.commit()
#             env.close()
#         self.open_lmdb_env()
#
#     def __len__(self):
#         # 以只读方式打开LMDB环境
#         assert self.env is not None
#         with self.env.begin(write=False) as txn:
#             # 获取样本数
#             byteflow = txn.get("__len__".encode())
#         if byteflow is not None:
#             length = pickle.loads(byteflow)
#             return length
#         else:
#             # 如果在数据库中没有找到样本数，返回0或抛出异常
#             return 0
#
#     # def process_one(self, row, lable_map):
#     #     reactant, product, label = row['Reactant'], row['Product'], row['Y']
#     #     catalyst = lable_map[label]
#     #     try:
#     #         r_graph = from_smiles(reactant)
#     #         p_graph = from_smiles(product)
#     #         c_graph = from_smiles(catalyst)
#     #     except ValueError as e:
#     #         # 处理异常情况，例如打印错误消息
#     #         print(f"Error processing data: {e}")
#     #         # 继续处理下一条数据
#     #         return None
#     #
#     #     return r_graph, p_graph, c_graph, label
#
#     def process_one(self, label, lable_map):
#         catalyst = lable_map[label]
#         try:
#             # r_graph = from_smiles(reactant)
#             # p_graph = from_smiles(product)
#             c_graph = from_smiles(catalyst)
#         except ValueError as e:
#             # 处理异常情况，例如打印错误消息
#             print(f"Error processing data: {e}")
#             # 继续处理下一条数据
#             return None
#
#         return c_graph, label
#
#     def __getitem__(self, idx):
#         assert self.env is not None
#         with self.env.begin(write=False) as txn:
#             byteflow = txn.get(f"{idx}".encode())
#         c_graph, label = pickle.loads(byteflow)
#         return c_graph, label
