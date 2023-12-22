from cmath import nan
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from models.Discriminator import Discriminator
from models.HGCN import HGCN
from models.HGCN2 import HGCN2
from models.EmbeddingLayer import UserEmbeddingLayer, ItemEmbeddingLayer, GroupEmbeddingLayer, AttentionLayer, PredictLayer


class HHGR(nn.Module):
    """
    HHGR framework for Group Recommendation:
    (a) User level hypergraph: 
    (b) Group level hypergraph: 
    """

    def __init__(self, n_items, n_users, n_groups, data_gu_dict, user_layers, lambda_mi=0.1, drop_ratio=0.4, aggregator_type='attention'):
        super(HHGR, self).__init__()
        self.n_items = n_items
        self.n_users = n_users
        self.n_groups = n_groups
        self.data_gu_dict = data_gu_dict
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)
        self.embedding_dim = user_layers[-1]
        self.aggregator_type = aggregator_type

        self.pre_userembedding = torch.tensor((n_users, self.embedding_dim))
        self.group_embeds = torch.Tensor()
        self.classifier = nn.Linear(1, 2)
        self.bias=0.5 
        self.classifier_threshold=0.5 
        self.large_penalty = 100

        ###############################################################################
        # Build the user level hypergraph
        ###############################################################################
        self.userembedds = UserEmbeddingLayer(self.n_users, self.embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(self.n_items, self.embedding_dim)
        self.hgcn_fine = HGCN2(
            self.embedding_dim, self.embedding_dim, self.embedding_dim, drop_ratio)
        self.hgcn_coarse = HGCN2(
            self.embedding_dim, self.embedding_dim, self.embedding_dim, drop_ratio)
        self.discriminator = Discriminator(embedding_dim=self.embedding_dim)

        ###############################################################################
        # Build the group level hypergraph
        ###############################################################################
        self.hgcn_gl = HGCN(self.embedding_dim,
                            self.embedding_dim, self.embedding_dim, drop_ratio)

        self.groupembeds = GroupEmbeddingLayer(
            self.n_groups, self.embedding_dim)
        self.attention = AttentionLayer(self.embedding_dim, drop_ratio)
        self.attention_pre = AttentionLayer(2 * self.embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * self.embedding_dim, drop_ratio)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def usr_forward(self, user_inputs, item_inputs):
        user_embeds = self.userembedds(torch.LongTensor(user_inputs))
        item_embeds = self.itemembeds(torch.LongTensor(item_inputs))
        element_embeds = torch.mul(
            user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat(
            (element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # def group_embed(self, group_inputs, user_embedding, H_gl_sp):
    #     group_embeds = torch.Tensor()
    #     print('the number of group inputs in group-level:', len(group_inputs))
    #     for i in group_inputs:
    #         if i % 100 == 0:
    #             print(i + 1)
    #         members = self.data_gu_dict[i]
    #         members_embeds = self.userembedds(torch.LongTensor(members)) # []
    #         for u in members:
    #             members_embeds[u] = user_embedding[u] + members_embeds[u]
    #    
    #         at_wt = self.attention(members_embeds)
    #         g_embeds_with_attention = torch.matmul(at_wt, members_embeds)

    #         group_embeds_pure = self.groupembeds(torch.LongTensor([i]))
    #         g_embeds = g_embeds_with_attention + group_embeds_pure
    #         group_embeds = torch.cat((group_embeds, g_embeds))
    #         del members, members_embeds, at_wt, g_embeds_with_attention, group_embeds_pure, g_embeds
    #     gc.collect()
    #     group_embeds_gl = self.hgcn_gl(group_embeds, H_gl_sp)
    #     return group_embeds_gl

    # group forward
    def train_grp_forward(self, group_inputs, item_inputs, user_embedding, group_embedds):
        group_embeds = torch.Tensor()
        group_types=torch.Tensor()
        group_weights=torch.Tensor()
        item_embeds_full = self.itemembeds(torch.LongTensor(item_inputs))
      
        for i, j in zip(group_inputs, item_inputs):
            members = self.data_gu_dict[i.item()]
            members = torch.LongTensor(members)
            members_embeds = self.userembedds(members)
            # members_embeds = members_embeds + user_embedding[members, :]
            items_numb = []
            for u in members:
                items_numb.append(j)
            item_embeds = self.itemembeds(torch.LongTensor(items_numb))
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)

            at_wt = self.attention_pre(group_item_embeds)
            #-------------------soft-------------------
            max_weightmemver_indices = torch.argmax(at_wt, dim=1)
            classification_scores = self.classifier(at_wt)  
            classification_scores[:,1]=classification_scores[:,1]+self.bias
            predicted_classes = torch.argmax(torch.softmax(classification_scores, dim=1), dim=1)  
            if predicted_classes==1:
                g_embeds_with_attention=members_embeds[max_weightmemver_indices]
            else:
                g_embeds_with_attention=torch.matmul(at_wt, members_embeds)
            # weight_fc=at_wt.var().item()
            # if weight_fc>threshold_mafengwo[3]:
            #     a = torch.argmax(at_wt)
            #     g_embeds_with_attention = members_embeds[a]
            # else:
            #     g_embeds_with_attention = torch.matmul(at_wt, members_embeds)

            # if at_wt[0].numel()==1 or at_wt[0].numel()==0:
            #     weight_fc.append(0)
            # else:
            #     weight_fc.append(at_wt.var().item())
            # g_embeds_with_attention = torch.matmul(
            #     at_wt, members_embeds)  
            
            device = torch.device("cuda")
            if len(at_wt) < target_size:
                num_pad_elements = target_size - len(at_wt[0])
                padding = torch.full((num_pad_elements,), float(0)).cuda(1)
                at_wt = torch.cat((at_wt[0], padding.to(device)), dim=0)
            group_types=torch.cat((group_types.to(device), predicted_classes))
            group_weights=torch.cat((group_weights.to(device), at_wt.unsqueeze(0)),dim=0)
           
            group_embeds_pure = self.groupembeds(torch.LongTensor([i]).to(device))
            group_embeds_pure = group_embeds_pure + group_embedds[[i], :]
            g_embeds = g_embeds_with_attention + group_embeds_pure  
            group_embeds = torch.cat((group_embeds.to(device), g_embeds))
            del members, members_embeds, item_embeds, group_item_embeds, at_wt, g_embeds_with_attention, group_embeds_pure, g_embeds
        gc.collect()
        element_embeds = torch.mul( 
            group_embeds, item_embeds_full)  # Element-wise product
        new_embeds = torch.cat(
            (element_embeds, group_embeds, item_embeds_full), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        
        return new_embeds, y,group_weights,group_types
    def weight_loss(self,weights,targets): 
        # weights = weights.squeeze(-1)
        leader_indices = torch.where(targets == 1)[0] 
        collaboration_indices = torch.where(targets == 0)[0]
        if len(leader_indices) == 0:  # If there are no leader groups, return a large penalty
            return self.large_penalty
        # Calculate indicators for leader groups
        leader_weights = weights[leader_indices]
        leader_indicators = self.calculate_indicator(leader_weights,1) 
        leader_indicators_repeated = leader_indicators.unsqueeze(1).expand(-1,5) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(0, collaboration_indices.size(0), (leader_indices.size(0), 5))
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
    def multinomial_loss(self, logits, items):
        """ multinomial likelihood with softmax over item set """
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def SSL_task(self, user_emb_coarse, user_emb_fine, user_neg_embedding):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(user_emb_fine, user_emb_coarse)  # (256, )
        neg1 = score(user_emb_coarse, user_neg_embedding)  # (256, )
        neg2 = score(user_emb_fine, user_neg_embedding)  # (256, )
        con_loss = -torch.sum(-torch.log(torch.sigmoid(pos)) -
                              torch.log(1-torch.sigmoid(neg1))-torch.log(1-torch.sigmoid(neg2)))
        return con_loss

    def SSL_loss(self, u_embeds_coarse, u_embeds_fine, u_embeds_neg):
        mi_loss = self.SSL_task(u_embeds_coarse, u_embeds_fine, u_embeds_neg)

        return mi_loss

    # def infomax_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
    #                        corrupted_user_items, device='cpu'):
    #     group_user_embeds = self.user_preference_encoder(
    #         user_items)  # [B, G, D]
    #     corrupt_user_embeds = self.user_preference_encoder(
    #         corrupted_user_items)  # [B, N, D]

    #     scores_observed = self.discriminator(  #
    #         group_embeds, group_user_embeds, group_mask)  # [B, G]
    #     scores_corrupted = self.discriminator(
    #         group_embeds, corrupt_user_embeds, group_mask)  # [B, N]

    #     mi_loss = self.discriminator.mi_loss(
    #         scores_observed, group_mask, scores_corrupted, device=device)

    #     ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
    #     user_items_norm = user_items / \
    #         torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
    #     gi_sum = group_items.sum(1, keepdim=True)
    #     group_items_norm = group_items / \
    #         torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
    #     assert scores_ug.requires_grad is False

    #     group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
    #     scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

    #     user_items_norm = torch.sum(
    #         user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
    #     user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
    #     group_loss = self.multinomial_loss(group_logits, group_items_norm)

    #     return mi_loss, user_group_loss, group_loss

    # def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
    #          device='cpu'):
    #     """ L_G + lambda L_UG + L_MI """
    #     mi_loss, user_group_loss, group_loss = self.infomax_group_loss(group_logits, summary_embeds, scores_ug,
    #                                                                    group_mask, group_items, user_items,
    #                                                                    corrupted_user_items, device)

    #     return group_loss + mi_loss + self.lambda_mi * user_group_loss
