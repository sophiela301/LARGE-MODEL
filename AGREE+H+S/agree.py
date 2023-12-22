'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''
import numpy as np
import torch
import torch.nn as nn


class AGREE(nn.Module):
    def __init__(self, num_users, num_items_u, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio, device):
        super(AGREE, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds_u = ItemEmbeddingLayer(num_items_u, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.device = device
        self.num_groups = len(self.group_member_dict)
        self.classifier = nn.Linear(1, 2)
        self.bias=0.5
        self.classifier_threshold=0.5
        self.large_penalty = 100 
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
        group_embeds = torch.Tensor()
        group_types=torch.Tensor()
        group_weights=torch.Tensor()
        item_embeds_full = self.itemembeds(
            torch.cuda.LongTensor(item_inputs))  
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i.item()]
            members = torch.cuda.LongTensor(members)#GPUtorch.cuda.LongTensor(members)
            members_embeds = self.userembeds(members)
            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_embeds = self.itemembeds(torch.cuda.LongTensor(items_numb))#GPU
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
            at_wt = self.attention(group_item_embeds)
        #     weight_fc=at_wt.var().item()
        #     if weight_fc>threshold_list_camra[1]:
        #         a = torch.argmax(at_wt)
        #         g_embeds_with_attention = members_embeds[a]
        #     else:
        #         g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
             #-------------------soft-------------------
            max_weightmemver_indices = torch.argmax(at_wt, dim=1)
            classification_scores = self.classifier(at_wt) 
            predicted_classes = torch.argmax(torch.softmax(classification_scores, dim=1), dim=1) 
            if predicted_classes==1:
                g_embeds_with_attention=members_embeds[max_weightmemver_indices]
            else:
                g_embeds_with_attention=torch.matmul(at_wt, members_embeds)
            target_size = 4 
            device = torch.device("cuda")
            if len(at_wt) < target_size:
                num_pad_elements = target_size - len(at_wt[0])
                padding = torch.full((num_pad_elements,), float(0)).cuda(1)
                at_wt = torch.cat((at_wt[0], padding.to(device)), dim=0)
            group_types=torch.cat((group_types.to(device), predicted_classes))
            group_weights=torch.cat((group_weights.to(device), at_wt.unsqueeze(0)),dim=0)
            
            group_embeds_pure = self.groupembeds(torch.cuda.LongTensor([i]))#GPU
            g_embeds = g_embeds_with_attention + group_embeds_pure
            group_embeds = torch.cat((group_embeds.to(self.device), g_embeds))

        element_embeds = torch.mul(
            group_embeds, item_embeds_full)  # Element-wise product   group[512, 64]
        new_embeds = torch.cat(
            (element_embeds, group_embeds, item_embeds_full), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        # weight_fc.sort(reverse = True)  #c
        # n_high_variances = int(0.5 * len(weight_fc))
        # threshold = weight_fc[n_high_variances-1]
        # print(threshold)
        return y,group_weights,group_types
    def weight_loss(self,weights,targets): 
        # weights = weights.squeeze(-1)
        leader_indices = torch.where(targets == 1)[0] 
        collaboration_indices = torch.where(targets == 0)[0]
        if len(leader_indices) == 0:  # If there are no leader groups, return a large penalty
            return self.large_penalty
        # Calculate indicators for leader groups
        leader_weights = weights[leader_indices]
        leader_indicators = self.calculate_indicator(leader_weights,1) 
        leader_indicators_repeated = leader_indicators.unsqueeze(1).expand(-1,3) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(0, collaboration_indices.size(0), (leader_indices.size(0), 3))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = self.calculate_indicator(selected_collaboration_weights,0)

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated), self.classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty
    def calculate_indicator(self,weights,isleader):
        rest_weights = weights.clone()
        rest_weights[weights == 0] = float('nan')  # Replace 0 weights with nan
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
    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        user_embeds = self.userembeds(user_inputs)
        item_embeds = self.itemembeds_u(item_inputs)
        element_embeds = torch.mul(user_embeds, item_embeds)
        new_embeds = torch.cat(
            (element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y,1,1


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs.cuda(0))# GPU
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
