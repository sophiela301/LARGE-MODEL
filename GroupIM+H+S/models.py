import torch.nn as nn
import torch.nn.functional as F
import torch
from models.aggregators import MaxPoolAggregator, AttentionAggregator, MeanPoolAggregator
from models.discriminator import Discriminator
from models.encoder import Encoder


class GroupIM(nn.Module):
    """
    GroupIM framework for Group Recommendation:
    (a) User Preference encoding: user_preference_encoder
    (b) Group Aggregator: preference_aggregator
    (c) InfoMax Discriminator: discriminator
    """

    def __init__(self, n_items, user_layers, lambda_mi=0.1, drop_ratio=0.4, aggregator_type='attention'):
        super(GroupIM, self).__init__()
        self.n_items = n_items
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)
        self.embedding_dim = user_layers[-1]
        self.aggregator_type = aggregator_type
        self.classifier_threshold=0.5
        self.large_penalty = 100
        self.user_preference_encoder = Encoder(
            self.n_items, user_layers, self.embedding_dim, drop_ratio)

        if self.aggregator_type == 'maxpool':
            self.preference_aggregator = MaxPoolAggregator(
                self.embedding_dim, self.embedding_dim)
        elif self.aggregator_type == 'meanpool':
            self.preference_aggregator = MeanPoolAggregator(
                self.embedding_dim, self.embedding_dim)
        elif self.aggregator_type == 'attention':
            self.preference_aggregator = AttentionAggregator(
                self.embedding_dim, self.embedding_dim)
        else:
            raise NotImplementedError(
                "Aggregator type {} not implemented ".format(self.aggregator_type))
       
        self.group_predictor = nn.Linear(
            self.embedding_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(
            self.group_predictor.weight) 

        self.discriminator = Discriminator(embedding_dim=self.embedding_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            # i += 1
            # print(i)

    def forward(self, group, group_users, group_mask, user_items):
        """ compute group embeddings and item recommendations by user preference encoding, group aggregation and
        item prediction

        :param group: [B] group id
        :param group_users: [B, G] group user ids with padding
        :param group_mask: [B, G] -inf/0 for absent/present user
        :param user_items: [B, G, I] individual item interactions of group members

        """
        user_pref_embeds = self.user_preference_encoder(user_items)  
        group_embed,weights,targets = self.preference_aggregator(
            user_pref_embeds, group_mask, mlp=False)  

        group_logits = self.group_predictor(group_embed)  
        user_logits = self.group_predictor(user_pref_embeds)  #[B,G,I]

        if self.train:
            obs_user_embeds = self.user_preference_encoder(
                user_items)  # [B, G, D]
            scores_ug = self.discriminator(
                group_embed, obs_user_embeds, group_mask).detach()  # [B, G]
            return group_logits, group_embed, scores_ug,weights,targets  
            # return group_logits, group_embed, scores_ug
        else:
            return group_logits, group_embed

    def multinomial_loss(self, logits, items):
        """  """
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))  

    def user_loss(self, user_logits, user_items):  # LU
        return self.multinomial_loss(user_logits, user_items)

    def infomax_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                           corrupted_user_items, device='cpu'):
        """ loss function with three terms: L_G, L_UG, L_MI
            :param group_logits: [B, G, I] group item predictions
            :param group_embeds: [B, D] group embedding
            :param scores_ug: [B, G] discriminator scores for group members
            :param group_mask: [B, G] -inf/0 for absent/present user
            :param group_items: [B, I] item interactions of group
            :param user_items: [B, G, I] individual item interactions of group members
            :param corrupted_user_items: [B, N, I] individual item interactions of negative user samples
            :param device: cpu/gpu
        """

        group_user_embeds = self.user_preference_encoder(
            user_items)  # [B, G, D]
        corrupt_user_embeds = self.user_preference_encoder(
            corrupted_user_items)  # [B, N, D]

        scores_observed = self.discriminator(
            group_embeds, group_user_embeds, group_mask)  # [B, G]
        scores_corrupted = self.discriminator(
            group_embeds, corrupt_user_embeds, group_mask)  # [B, N]

        mi_loss = self.discriminator.mi_loss(
            scores_observed, group_mask, scores_corrupted, device=device)

        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / \
            torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / \
            torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        assert scores_ug.requires_grad is False

        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

        user_items_norm = torch.sum(
            user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(
            group_logits, user_items_norm)  # LUG
        group_loss = self.multinomial_loss(
            group_logits, group_items_norm)  # LG

        return mi_loss, user_group_loss, group_loss

    def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
             device='cpu'):
        """ L_G + lambda L_UG + L_MI """
        mi_loss, user_group_loss, group_loss = self.infomax_group_loss(group_logits, summary_embeds, scores_ug,
                                                                       group_mask, group_items, user_items,
                                                                       corrupted_user_items, device)

        return group_loss + mi_loss + self.lambda_mi * user_group_loss
    
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
        leader_indicators_repeated = leader_indicators.unsqueeze(1).expand(-1,4) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(0, collaboration_indices.size(0), (leader_indices.size(0), 4))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = self.calculate_indicator(selected_collaboration_weights,0)

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated), self.classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty

    