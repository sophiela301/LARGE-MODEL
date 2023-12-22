import torch
import torch.nn as nn
import numpy as np
import random
from dgl import mean_nodes
from dgl.nn.pytorch import GraphConv, GINConv, RelGraphConv
# from dgl.nn.pytorch.glob import AvgPooling
import dgl.nn as dglnn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from dgl.nn.pytorch.glob import AvgPooling
import torch.nn.functional as F


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=256):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.Tanh(),
            nn.Linear(in_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        betas = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)        ###

        return beta, (betas * z).sum(1)  # (N, D * K)      ###


class GROUP_single(nn.Module):

    def __init__(self, args):
        super(GROUP_single, self).__init__()
        self.args = args
        self.n_groups = args.n_node['group']
        self.n_users = args.n_node['user']
        self.n_items = args.n_node['item']
        self.embeds_item_last = torch.zeros(1, args.hid_dim).to(args.device)

        self.group_embedding = torch.nn.Embedding(
            num_embeddings=self.n_groups, embedding_dim=args.hid_dim)
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=args.hid_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=args.hid_dim)

        self.graphconv = dglnn.GraphConv(args.hid_dim, args.hid_dim, norm='both', weight=False, bias=False,
                                         activation=None)

        self.semantic_attention_group = SemanticAttention(in_size=args.hid_dim)
        self.semantic_attention_user = SemanticAttention(in_size=args.hid_dim)
        self.semantic_attention_item = SemanticAttention(in_size=args.hid_dim)

        self.r0 = nn.Parameter(torch.randn(args.hid_dim, 1))
        self.dropout = nn.Dropout(args.drop_out)
        self.apply(xavier_uniform_initialization)
        self.classifier_threshold=0.5 
        self.large_penalty = 100 

    def get_embedding(self, g_all_emb, u_all_emb, i_all_emb):
        semantic_group = torch.stack(g_all_emb, dim=1)
        semantic_user = torch.stack(u_all_emb, dim=1)
        semantic_item = torch.stack(i_all_emb, dim=1)
        beta1, h_group = self.semantic_attention_group(semantic_group)
        beta2, h_user = self.semantic_attention_user(semantic_user)
        beta3, h_item = self.semantic_attention_item(semantic_item)
        h_item = torch.cat((h_item, self.embeds_item_last.data), 0)

        return h_group, h_user, h_item
   

    def forward(self, graphs):
        g_all_emb = list()
        u_all_emb = list()
        i_all_emb = list()

        group_embeddings = self.group_embedding.weight
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        gi_emb = torch.cat([group_embeddings, item_embeddings], dim=0)
        ui_emb = torch.cat([user_embeddings, item_embeddings], dim=0)
        gu_emb = torch.cat([group_embeddings, user_embeddings], dim=0)

        # for i in range(self.args.layers):
        #     gu_emb = self.graphconv(graphs['GU'], gu_emb)
        # group_h, user_h = torch.split(gu_emb, [self.n_groups, self.n_users])
        # g_all_emb.append(group_h.flatten(1))
        # u_all_emb.append(user_h.flatten(1))

        for i in range(self.args.layers):
            ui_emb = self.graphconv(graphs['UI'], ui_emb)
        user_h, item_h = torch.split(ui_emb, [self.n_users, self.n_items])
        u_all_emb.append(user_h.flatten(1))
        i_all_emb.append(item_h.flatten(1))

        for i in range(self.args.layers):
            gi_emb = self.graphconv(graphs['GI'], gi_emb)
        group_h, item_h = torch.split(gi_emb, [self.n_groups, self.n_items])
        g_all_emb.append(group_h.flatten(1))
        # i_all_emb.append(item_h.flatten(1))

        self.h_group, self.h_user, self.h_item = g_all_emb[0], u_all_emb[0], i_all_emb[0]  #实验
        self.h_item = torch.cat((self.h_item, self.embeds_item_last.data), 0)
        
        return self.h_group, self.h_user, self.h_item

    def loss(self, h_group, h_user, h_item, gids, g_iid_list, uids, u_iid_list):
        h_group = self.dropout(h_group[gids])
        loss0 = self.get_loss(h_group, h_item, g_iid_list,
                              self.r0, self.args.neg_weight['group'])
        return loss0

    def loss1(self, h_group, h_user, h_item, gids, g_iid_list, uids, u_iid_list):
        h_user = self.dropout(h_user[uids])
        loss0 = self.get_loss(h_user, h_item, u_iid_list,
                              self.r0, self.args.neg_weight['user'])
        return loss0

    # def rank(self, h_user0, h_item0):

    #     user_all_items = h_user0.unsqueeze(1) * h_item0  # batchsize,item,64
    #     items_score0 = user_all_items.matmul(self.r0).squeeze(2)
    #     return items_score0
    def calculate_indicator(self,weights,isleader):
        rest_weights = weights.clone()
        rest_weights[weights == 0] = float('nan')  
        if isleader==1:
            max_index = torch.argmax(weights,dim=1)
            max_weight = weights[torch.arange(weights.size(0)), max_index]

            rest_weights = torch.cat([weights[:,:max_index[0]], weights[:,max_index[0]+1:]], dim=1)

            
            weights_clone = rest_weights.clone()
            weights_clone = torch.where(weights_clone == 0, torch.tensor(float('nan')).cuda(), weights_clone)
            mask = torch.isnan(weights_clone)
            rest_mean = torch.sum(weights_clone.masked_fill_(mask, 0.), dim=1) / mask.logical_not().sum(dim=1)
            indicator = (max_weight - rest_mean) / rest_mean
        else:
            max_index = torch.argmax(weights, dim=2) # Find index of maximum weight within each group
            max_weight = weights.gather(2, max_index.unsqueeze(2)).squeeze(2)   
            weights_clone = weights.clone()
            max_index_exp = max_index.unsqueeze(2)
            # Set the max weights and 0 values to nan
            weights_clone = weights_clone.scatter_(2, max_index_exp, float('nan'))
            weights_clone = torch.where(weights_clone == 0, torch.tensor(float('nan')).cuda(), weights_clone)
            # Mask where nan values
            mask = torch.isnan(weights_clone)
            # Calculate mean value while ignoring nan
            rest_mean = torch.sum(weights_clone.masked_fill_(mask, 0.), dim=2) / mask.logical_not().sum(dim=2)

            indicator = (max_weight - rest_mean) / rest_mean
        return indicator
    def weight_loss(self,weights,targets):
        weights = weights.squeeze(-1)
        leader_indices = torch.where(targets == 1)[0] 
        collaboration_indices = torch.where(targets == 0)[0]
        if len(leader_indices) == 0:  # If there are no leader groups, return a large penalty
            return self.large_penalty
        # Calculate indicators for leader groups
        leader_weights = weights[leader_indices]
        leader_indicators = self.calculate_indicator(leader_weights,1) 
        leader_indicators_repeated = leader_indicators.unsqueeze(1).expand(-1,2) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(0, collaboration_indices.size(0), (leader_indices.size(0), 2))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = self.calculate_indicator(selected_collaboration_weights,0) 

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated), self.classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty
    def rank(self, h_user0, h_item0):
        items_score = torch.zeros(h_user0.shape[0], h_item0.shape[0])

        X = zip(h_user0)
        for i, x in enumerate(X):
            x = x[0].unsqueeze(0)
            user_all_items = x.unsqueeze(1) * h_item0
            items_score0 = user_all_items.matmul(self.r0).squeeze(2)
            items_score[i] = items_score0

        return items_score

    def get_loss(self, h_u, h_i, pos_iids, r, neg_weight):
        item_num = h_i.shape[0] - 1
        mask = (~(pos_iids.eq(item_num))).float()
        pos_embs = h_i[pos_iids]
        pos_embs = pos_embs * mask.unsqueeze(2)  # 512,5,64
        pq = h_u.unsqueeze(1) * pos_embs
        hpq = pq.matmul(r).squeeze(2)
        pos_data_loss = torch.sum((1 - neg_weight) * hpq.square() - 2.0 * hpq)
        part_1 = h_u.unsqueeze(2).bmm(h_u.unsqueeze(1))
        part_2 = h_i.unsqueeze(2).bmm(h_i.unsqueeze(1))

        part_1 = part_1.sum(0)
        part_2 = part_2.sum(0)
        part_3 = r.mm(r.t())

        all_data_loss = torch.sum(part_1 * part_2 * part_3)
        loss = neg_weight * all_data_loss + pos_data_loss

        return loss


def xavier_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
