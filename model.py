import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import HypergraphConv
from torch.nn.utils.rnn import pack_padded_sequence

    

class HyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(HyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.hconv_layer = HypergraphConv(in_channels=emb_dim,out_channels=emb_dim,use_attention=True,dropout=dropout)
        # self.hconv_layer = HypergraphConv(in_channels=emb_dim,out_channels=emb_dim,use_attention=False,dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pois_embs, H,W,traj_embs):
        # traj-poi matrix poi-traj matrix
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.hconv_layer(pois_embs, H,W,traj_embs)  # [L, d]
            # pois_embs = self.hconv_layer(pois_embs, H,W)  # [L, d]
            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            # pois_embs = self.dropout(pois_embs)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class HGTUL(nn.Module):
    def __init__(self, num_trajs, num_pois, args, device):
        super(HGTUL, self).__init__()

        # definition
        self.num_trajs = num_trajs
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        # embedding
        self.traj_embedding = nn.Embedding(num_trajs, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # embedding init
        nn.init.xavier_uniform_(self.traj_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # network
        self.hconv_network = HyperConvNetwork(args.num_layers, args.emb_dim,args.dropout, device)

        # gating before disentangled learning
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        self.hour_time_emb = nn.Embedding(48,self.emb_dim)
        self.weeday_emb = nn.Embedding(2,self.emb_dim)
        self.geo_emb = nn.Embedding(args.n_all_geoids,self.emb_dim)
        self.lstm = nn.LSTM(args.emb_dim, args.emb_dim, batch_first=True)
        self.classify = nn.Linear(args.emb_dim,args.user_number)

    def forward(self, dataset,traj_data):

        # multi-view hypergraph convolutional network
        H = dataset['H'].to(self.device)
        # W = dataset["W"].to(self.device)
        W =None
        hg_pois_embs = self.hconv_network(self.poi_embedding.weight[:-1], H,W,self.traj_embedding.weight)
        # hypergraph structure aware users embeddings
        H_matrix = dataset['H_matrix'].to(self.device)
        hg_structural_traj_embs = torch.sparse.mm(H_matrix,hg_pois_embs)  # [U, d]
        # hg_batch_trajs_embs = hg_structural_traj_embs
        hg_batch_trajs_embs = hg_structural_traj_embs+self.traj_embedding.weight
        # hg_batch_trajs_embs = self.traj_embedding.weight

        
        # normalization
        # norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_hg_batch_trajs_embs = F.normalize(hg_batch_trajs_embs, p=2, dim=1)


        padded_trajs = traj_data['trajs'].to(self.device)
        padded_trajs_hour = traj_data['trajs_hour'].to(self.device)
        padded_trajs_week = traj_data['trajs_week'].to(self.device)
        trajs_len = traj_data['trajs_len']
        padded_trajs = self.geo_emb(padded_trajs)
        padded_trajs_hour = self.hour_time_emb(padded_trajs_hour)
        padded_trajs_week = self.weeday_emb(padded_trajs_week)
        padded_emb = padded_trajs+padded_trajs_week+padded_trajs_hour
        pack_padded_emb = pack_padded_sequence(padded_emb, trajs_len, batch_first=True, enforce_sorted=False)
        _, (sequential_traj_embs, _) = self.lstm(pack_padded_emb)
        sequential_traj_embs = sequential_traj_embs.transpose(0,1).reshape(padded_trajs.shape[0], -1)
        norm_sequential_traj_embs = F.normalize(sequential_traj_embs,p=2,dim=1)
        # prediction
        prediction = self.classify(norm_hg_batch_trajs_embs+norm_sequential_traj_embs)
        # prediction = self.classify(norm_sequential_traj_embs)
        # prediction = self.classify(norm_hg_batch_trajs_embs)
        return prediction



