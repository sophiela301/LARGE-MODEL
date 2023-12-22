import torch.nn as nn
import torch
from collections import defaultdict
import numpy as np

class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.linear(x)


class OverlapGraphConvolution(nn.Module):
    """Graph Convolution for Group-level graph"""

    def __init__(self, layers):
        super(OverlapGraphConvolution, self).__init__()
        self.layers = layers

    def forward(self, embedding, adj):
        group_emb = embedding
        final = [group_emb]

        for _ in range(self.layers):
            group_emb = torch.mm(adj, group_emb)
            final.append(group_emb)

        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


class LightGCN(nn.Module):
    """Graph Convolution for Item-level graph"""

    def __init__(self, num_groups, num_items, layers, g):
        super(LightGCN, self).__init__()

        self.num_groups, self.num_items = num_groups, num_items
        self.layers = layers
        self.graph = g

    def compute(self, groups_emb, items_emb):
        """LightGCN forward propagation"""
        all_emb = torch.cat([groups_emb, items_emb])
        embeddings = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embeddings.append(all_emb)
        embeddings = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        groups, _ = torch.split(embeddings, [self.num_groups, self.num_items])
        return groups

    def forward(self, groups_emb, items_emb):
        return self.compute(groups_emb, items_emb)


class HyperGraphBasicConvolution(nn.Module):
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(3 * input_dim, input_dim)
        
        
        self.classifier = nn.Linear(1, 2).cuda(1)
        self.bias=-0.73 #给leader组的bias0.87-0.8849-0.884808

   

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper,weight,group_member_embeddings): 
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)

        item_group_element = item_msg * group_emb
        
        
        # low_variance_group_embedding=user_msg[low_variance_indices] 
        # for i in range(low_variance_indices.shape[0]): 
        #     ret[low_variance_indices[i]] = low_variance_group_embedding[i]
        max_weightmemver_indices = torch.argmax(weight, dim=1)
        classification_scores = self.classifier(weight.squeeze(2))  
        classification_scores[:,1]=classification_scores[:,1]+self.bias
        predicted_classes = torch.argmax(torch.softmax(classification_scores, dim=1), dim=1) 

        ret=torch.zeros(group_member_embeddings.shape[0],group_member_embeddings.shape[2])
        for i in range(len(predicted_classes)):
            if predicted_classes[i] ==1:#leadership
                ret[i]=group_member_embeddings[i][max_weightmemver_indices[i]]
            else:
                ret[i] = user_msg[i] 
        msg = self.aggregation(torch.cat([ret.cuda(1), item_msg, item_group_element], dim=1))
        norm_emb = torch.mm(full_hyper, msg)
        return norm_emb, msg,predicted_classes 


class HyperGraphConvolution(nn.Module):
    """Hyper-graph Convolution for Member-level hyper-graph"""

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]
        self.group_member_dict = self.load_group_member_to_dict(
            f"./data/C622/groupMember.txt")
        self.attention = nn.Linear(input_dim, 1)
    def load_group_member_to_dict(self,user_in_group_path):
        """Return **Dict** format group-to-member-list mapping"""
        group_member_dict = defaultdict(list)
        lines = open(user_in_group_path, 'r').readlines()

        for line in lines:
            contents = line.split()
            group = int(contents[0])
            for member in contents[1].split(','):
                group_member_dict[group].append(int(member))
        return group_member_dict

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        final = [torch.cat([user_emb, item_emb], dim=0)]
        final_he = [group_emb]
        max_members = max(len(members) for members in self.group_member_dict.values())
        group_member_embeddings = torch.zeros(290, max_members, 32)
        group_mask = torch.full((290, max_members), float('-inf'))
        for group_id, members in self.group_member_dict.items():
            for i, member_id in enumerate(members):
                group_member_embeddings[group_id, i] = user_emb[member_id]  
            group_mask[group_id, :len(members)] = 0
        group_mask=group_mask.cuda(1)
        attention_out = torch.tanh(self.attention(group_member_embeddings.cuda(1)))
        weight = torch.softmax(attention_out + group_mask.unsqueeze(2), dim=1)
        
        # variances=torch.zeros(weight.shape[0],1)
        # for i in range(weight.shape[0]):
        #     list_member = []
        #     for j in range(weight.shape[1]):
        #         if weight[i][j] !=0.0:
        #             list_member.append(weight[i][j].cpu().detach().numpy())
        #     variances[i]= torch.tensor(np.var(list_member,ddof = 1))  
        # #Find groups with high variance and return the average of top k weighted members
        # ret=torch.zeros(group_member_embeddings.shape[0],group_member_embeddings.shape[2])
        # sorted_variances, _ = torch.sort(variances,descending=True,dim=0)
        # n_high_variances = int(0.05 * variances.shape[0])
        # threshold = sorted_variances[n_high_variances-1]

        # high_variance_mask = (variances > threshold).float()  
        # high_variance_indices = high_variance_mask.nonzero(as_tuple=True)[0]  
        # high_variance_members = torch.topk(weight[high_variance_indices], k=1, dim=1).indices.squeeze()  
        # high_variance_member_embedding=group_member_embeddings[high_variance_indices] 
        # # high_variance_group_embedding = torch.zeros(high_variance_members.shape[0],32)
        # for i in range(high_variance_indices.shape[0]):
        #     if high_variance_indices.shape[0]==1:
        #         ret[high_variance_indices[i]]=high_variance_member_embedding[i][high_variance_members]
        #     else:
        #         ret[high_variance_indices[i]]=high_variance_member_embedding[i][high_variance_members[i]]
        # low_variance_mask = (variances <= threshold).float()
        # low_variance_indices = low_variance_mask.nonzero(as_tuple=True)[0]
        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg,target = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph,weight,group_member_embeddings)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final.append(emb)
            final_he.append(he_msg)

        final_emb = torch.sum(torch.stack(final), dim=0)
        final_he = torch.sum(torch.stack(final_he), dim=0)
        return final_emb, final_he,weight,target 


class ConsRec(nn.Module):
    """ConsRec: Consensus-based Group Recommender"""

    def __init__(self, num_users, num_items, num_groups, args, user_hyper_graph, item_hyper_graph,
                 full_hyper, overlap_graph, device, light_gcn_graph, num_lgcn_item):
        super(ConsRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        # Hyper-parameters
        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        self.overlap_graph = overlap_graph
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        # Embedding init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        # Gate
        self.overlap_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.lightgcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Hyper-graph Convolution
        self.hyper_graph_conv = HyperGraphConvolution(user_hyper_graph, item_hyper_graph, full_hyper, self.layers,
                                                      self.emb_dim, device)

        # Overlap-graph Convolution
        self.overlap_graph_conv = OverlapGraphConvolution(self.layers)

        # LightGCN Convolution
        self.light_gcn = LightGCN(num_groups, num_lgcn_item, self.layers, light_gcn_graph)
        self.num_lgcn_item = num_lgcn_item

        # Prediction Layer
        self.predict = PredictLayer(self.emb_dim)
    
    def forward(self, group_inputs, user_inputs, item_inputs):
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, item_inputs)
        else:
            return self.user_forward(user_inputs, item_inputs)

    def group_forward(self, group_inputs, item_inputs):
        # Group-level graph computation
        group_emb = self.overlap_graph_conv(self.group_embedding.weight, self.overlap_graph)

        # Member-level graph computation
        ui_emb, he_emb,weight,target = self.hyper_graph_conv(self.user_embedding.weight, self.item_embedding.weight,
                                               group_emb,
                                               self.num_users, self.num_items)
        _, i_emb = torch.split(ui_emb, [self.num_users, self.num_items])

        # Item-level graph computation
        light_gcn_group_emb = self.light_gcn(self.group_embedding.weight,
                                             self.item_embedding.weight[:self.num_lgcn_item, :])

        # Combination
        overlap_coef, hyper_coef, lightgcn_coef = self.overlap_gate(group_emb), self.hyper_gate(
            he_emb), self.lightgcn_gate(light_gcn_group_emb)

        group_ui_emb = overlap_coef * group_emb + hyper_coef * he_emb + lightgcn_coef * light_gcn_group_emb
        i_emb = i_emb[item_inputs]
        g_emb = group_ui_emb[group_inputs]

        # For CAMRa2011, we use DOT mode to avoid the dead ReLU
        if self.predictor_type == "MLP":
            # return torch.sigmoid(self.predict(g_emb * i_emb))
            return torch.sigmoid(self.predict(g_emb * i_emb)),weight,target #soft
        else:
            return torch.sum(g_emb * i_emb, dim=-1)

    def user_forward(self, user_inputs, item_inputs):
        u_emb = self.user_embedding(user_inputs)
        i_emb = self.item_embedding(item_inputs)
        if self.predictor_type == "MLP":
            return torch.sigmoid(self.predict(u_emb * i_emb))
        else:
            return torch.sum(u_emb * i_emb, dim=-1)
