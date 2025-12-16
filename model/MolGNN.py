import torch
from torch_geometric.nn import GAT, GraphSAGE, GIN, GCN, pool, global_mean_pool
from torch_geometric.utils.smiles import x_map, e_map
from torch_geometric.data import Data


class MolGNNLayers(torch.nn.Module):
    def __init__(self, d_model, heads=None, dropout=0.4, num_layers=4, mol_encoder='gnn_gat_v1'):
        super(MolGNNLayers, self).__init__()
        self.embedding_dim = d_model
        self.atom_encoder = AtomEncoder(d_model)
        self.bond_encoder = BondEncoder(d_model)
        self.mol_encoder = mol_encoder

        if mol_encoder == 'gnn_gat_v2':
            self.gnn = GAT(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channels=d_model,
                           dropout=dropout,
                           v2=True,
                           heads=heads,
                           edge_dim=d_model,
                           )
        elif mol_encoder == 'gnn_gat_v1':
            self.gnn = GAT(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channels=d_model,
                           dropout=dropout,
                           v2=False,
                           heads=heads,
                           edge_dim=d_model,
                           )
        elif mol_encoder == 'gnn_gcn':
            self.gnn = GCN(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channels=d_model,
                           dropout=dropout,
                           )
        elif mol_encoder == 'gnn_gin':
            self.gnn = GIN(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channels=d_model,
                           dropout=dropout,
                           )
        elif mol_encoder == 'gnn_graphsage':
            self.gnn = GraphSAGE(in_channels=d_model,
                                 hidden_channels=d_model,
                                 num_layers=num_layers,
                                 out_channels=d_model,
                                 dropout=dropout,
                                 )
        else:
            raise NotImplementedError('mol_encoder {} not implemented'.format(mol_encoder))

        self.fc = FullConnected(d_model, d_model, dropout=dropout)

    def forward(self, batch):
        """
        :param batch:  batch.x: atom features, shape (num_nodes, num_atom_fea)
        :return: mol embeddings, shape (batch_size, num_mols, embedding_dim)
        """
        x = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)
        if self.mol_encoder in ['gnn_gat_v1', 'gnn_gat_v2']:
            x = self.gnn(x, edge_index=batch.edge_index, edge_attr=edge_attr)
        else:
            x = self.gnn(x, edge_index=batch.edge_index)
        x = global_mean_pool(x, batch.batch)
        x = self.fc(x)
        return x


class AtomEncoder(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for key, value in x_map.items():
            emb = torch.nn.Embedding(len(value), embedding_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """
        :param x: feature matrix of atoms, shape (num_batched_atoms, num_atom_fea)
        :return: atom embeddings, shape (num_batched_atoms, embedding_dim)
        """
        encoded_features = 0
        for i in range(x.shape[1]):
            # if x[:, i].max() >= self.atom_embedding_list[i].weight.shape[0]:
            #     print(x[:, i].max(), self.atom_embedding_list[i].weight.shape[0])
            if x[:, i].max() >= self.atom_embedding_list[i].weight.shape[0]:
                raise ValueError(
                    f'Index {x[:, i].max()} out of range for embedding {i} with size {self.atom_embedding_list[i].weight.shape[0]}')
            encoded_features += self.atom_embedding_list[i](x[:, i])
        x = encoded_features
        return x


class BondEncoder(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding = torch.nn.ModuleList()
        for key, value in e_map.items():
            emb = torch.nn.Embedding(len(value), embedding_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding.append(emb)

    def forward(self, edge_attr):
        """
        :param edge_attr: edge attributes, shape (num_batched_edges, num_bond_fea)
        :return: bond embeddings, shape (num_batched_edges, embedding_dim)
        """
        x = 0
        for idx, emb in enumerate(self.bond_embedding):
            # 添加越界检查
            if (edge_attr[:, idx].max() >= emb.num_embeddings).any():
                raise ValueError(f"Edge attribute {idx} value out of range")
            x += emb(edge_attr[:, idx].long())

        return x


class FullConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.4):
        super(FullConnected, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.layer_norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.layer_norm(self.fc(x)))
