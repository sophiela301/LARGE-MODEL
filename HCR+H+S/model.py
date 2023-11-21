import torch
import torch.nn as nn
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(3)
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HGR(nn.Module):
    def __init__(self, num_users, num_items, num_groups, emb_dim, layers, drop_ratio, adj, D, A, group_member_dict, device):
        super(HGR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.layers = layers
        self.drop_ratio = drop_ratio
        self.adj = adj
        self.D = D
        self.A = A
        self.group_member_dict = group_member_dict
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)
        self.hyper_graph = HyperConv(self.layers)
        self.group_graph = GroupConv(self.layers)
        self.attention = AttentionLayer(2 * self.emb_dim, self.drop_ratio)
        self.predict = PredictLayer(3 * self.emb_dim, self.drop_ratio)
        self.device = device
        self.classifier = nn.Linear(1, 2)
        self.bias=-0.8 

        self.gate = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), nn.Sigmoid())

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)
        self.classifier_threshold=0.5
        self.large_penalty = 100 


    def forward(self, group_inputs, user_inputs, item_inputs):


        if (group_inputs is not None) and (user_inputs is None):
            ui_embedding = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            ui_embedding = self.hyper_graph(self.adj, ui_embedding)
            # user_embedding, item_embedding = torch.split(ui_embedding, [self.num_users, self.num_items], dim=0)
            user_embedding, item_embedding = torch.split(torch.tensor(ui_embedding), [self.num_users, self.num_items], dim=0)
            item_emb = item_embedding[item_inputs]

            group_embedding = self.group_graph(self.group_embedding.weight, self.D, self.A)  

            member = []
            max_len = 0
            bsz = group_inputs.shape[0]
            member_masked = []
            for i in range(bsz):
                member.append(np.array(self.group_member_dict[group_inputs[i].item()]))
                max_len = max(max_len, len(self.group_member_dict[group_inputs[i].item()]))
            mask = np.zeros((bsz, max_len))
            for i, item in enumerate(member):
                cur_len = item.shape[0]
                member_masked.append(np.append(item, np.zeros(max_len - cur_len)))
                mask[i, cur_len:] = 1.0
            member_masked = torch.LongTensor(member_masked).to(self.device) 
            mask = torch.Tensor(mask).to(self.device)  # 

            member_emb = user_embedding[member_masked]  
            x = member_emb.cuda(1)
            # attention aggregation
            item_emb_attn = item_emb.unsqueeze(1).expand(bsz, max_len, -1) 
            at_emb = torch.cat((member_emb, item_emb_attn), dim=2)

            at_wt = self.attention(at_emb, mask)  
            weight = at_wt.unsqueeze(2)
            
            max_weightmemver_indices = torch.argmax(weight, dim=1)
            classification_scores = self.classifier(weight.squeeze(2))  
            classification_scores[:,1]=classification_scores[:,1]+self.bias
            predicted_classes = torch.argmax(torch.softmax(classification_scores, dim=1), dim=1) 
            high_variance_indices=[]
            for i in range(len(predicted_classes)):
                if predicted_classes[i] ==1:#leadership
                    ret[i]=x[i][max_weightmemver_indices[i]]
                    high_variance_indices.append(i)
                else:
                    ret[i] = torch.sum(weight[i] * x[i].cuda(1), dim=0)

        

            g_emb_pure = torch.tensor(group_embedding)[group_inputs].cuda(1) 
            group_emb = g_emb_with_attention + g_emb_pure
           

            element_emb = torch.mul(group_emb, item_emb.cuda(1))# GPU)

            new_emb = torch.cat((element_emb, group_emb, item_emb.cuda(1)), dim=1).cuda(1)# GPU)
            y = torch.sigmoid(self.predict(new_emb))
            # y = torch.matmul(group_emb.unsqueeze(1), item_emb.unsqueeze(2)).squeeze()
            return y,weight,predicted_classes

        else:
            user_emb = self.user_embedding(user_inputs)
            item_emb = self.item_embedding(item_inputs)
            element_emb = torch.mul(user_emb, item_emb)
            new_emb = torch.cat((element_emb, user_emb, item_emb), dim=1)
            y = torch.sigmoid(self.predict(new_emb))
            # y = torch.matmul(user_emb.unsqueeze(1), item_emb.unsqueeze(2)).squeeze()
            return y,0,0
    def calculate_indicator(self,weights,isleader):
        rest_weights = weights.clone()
        rest_weights[weights == 0] = float('nan')  
        if isleader==1:
            max_index = torch.argmax(weights,dim=1)
            max_weight = weights[torch.arange(weights.size(0)), max_index]

            rest_weights = torch.cat([weights[:,:max_index[0]], weights[:,max_index[0]+1:]], dim=1)

            weights_clone = rest_weights.clone()
            weights_clone = torch.where(weights_clone == 0, torch.tensor(float('nan')).cuda(1), weights_clone)
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
            weights_clone = torch.where(weights_clone == 0, torch.tensor(float('nan')).cuda(1), weights_clone)
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
        leader_indicators_repeated = leader_indicators.unsqueeze(1).expand(-1,4) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(0, collaboration_indices.size(0), (leader_indices.size(0), 4))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = self.calculate_indicator(selected_collaboration_weights,0) 

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated), self.classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty
class HyperConv(nn.Module):
    def __init__(self, layers):
        super(HyperConv, self).__init__()
        self.layers = layers

    def forward(self, adj, embedding):
        all_emb = embedding
        final = [all_emb]
        for i in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            final.append(all_emb)
        final_emb = np.sum([t.cpu().detach().numpy() for t in final], axis=0)  # gpu
        # final_emb = np.sum([t.detach().numpy() for t in final], axis=0) #cpu
        return final_emb

class GroupConv(nn.Module):
    def __init__(self, layers):
        super(GroupConv, self).__init__()
        self.layers = layers

    def forward(self, embedding, D, A):
        DA = torch.mm(D, A).float()
        group_emb = embedding
        final = [group_emb]
        for i in range(self.layers):
            group_emb = torch.mm(DA, group_emb)
            final.append(group_emb)
            
        # final_emb = np.sum(final, 0) 
        final_emb = np.sum([t.cpu().detach().numpy() for t in final], axis=0)  # gpu
        # final_emb = np.sum([t.detach().numpy() for t in final], axis=0) #cpu
        return final_emb

class AttentionLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim / 2)),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(int(emb_dim / 2), 1)
        )

    def forward(self, x, mask):
        bsz = x.shape[0]
        # out = self.linear(x)  #CPU
        out = self.linear(x.cuda(1))#gPU
        out = out.view(bsz, -1) # [bsz, max_len]
        out.masked_fill_(mask.bool(), -np.inf)
        weight = torch.softmax(out, dim=1)
        return weight

class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

