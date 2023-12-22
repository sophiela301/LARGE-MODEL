import torch
import torch.nn as nn
import numpy as np


class MaxPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as max pooling over group member embeddings 
     max pooling"""

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MaxPoolAggregator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ max pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before max pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        if mask is None:
            return torch.max(h, dim=1)
        else:
            res = torch.max(h + mask.unsqueeze(2), dim=1)
            return res.values


# mask:  -inf/0 for absent/present.
class MeanPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as mean pooling over group member embeddings 
    mean pooling"""

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MeanPoolAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=False):
        """ mean pooling aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before mean pooling
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x
        if mask is None:
            return torch.mean(h, dim=1)
        else:
            mask = torch.exp(mask)
            res = torch.sum(h * mask.unsqueeze(2), dim=1) / \
                mask.sum(1).unsqueeze(1)
            return res


class AttentionAggregator(nn.Module):
    """ Group Preference Aggregator implemented as attention over group member embeddings
    attention"""

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(AttentionAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        self.attention = nn.Linear(output_dim, 1).cuda(1)
        self.drop = nn.Dropout(drop_ratio).cuda(1)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)
        self.classifier = nn.Linear(1, 2).cuda(1)
        # self.bias=-0.5 #

    def forward(self, x, mask, variance,mlp=False):
        """ attentive aggregator:
            :param x: [B, G, D]  group member embeddings
            :param mask: [B, G]  -inf/0 for absent/present
            :param mlp: flag to add a linear layer before attention
        """
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x

        attention_out = torch.tanh(self.attention(h))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        # 
        variances=torch.zeros(weight.shape[0],1)
        for i in range(weight.shape[0]):
            list_member = []
            for j in range(weight.shape[1]):
                if weight[i][j] !=0.0:
                    list_member.append(weight[i][j].cpu().detach().numpy())
            variances[i]= torch.tensor(np.var(list_member,ddof = 1))  # 
        # Find groups with high variance and return the average of top k weighted members
        # hard----------------------------------------------------
        ret=torch.zeros(x.shape[0],x.shape[2])
        sorted_variances, _ = torch.sort(variances,descending=True,dim=0)
        n_high_variances = int(variance * variances.shape[0])
        threshold = sorted_variances[n_high_variances-1]

        high_variance_mask = (variances > threshold).float()  # 
        high_variance_indices = high_variance_mask.nonzero(as_tuple=True)[0]  #
        high_variance_members = torch.topk(weight[high_variance_indices], k=1, dim=1).indices.squeeze()  # 
        high_variance_member_embedding=h[high_variance_indices]  #[k,mask,64]
        # high_variance_group_embedding = torch.zeros(high_variance_members.shape[0],64)
        for i in range(high_variance_indices.shape[0]):
            if high_variance_indices.shape[0]==1:
                ret[high_variance_indices[i]]=high_variance_member_embedding[i][high_variance_members]
            else:
                ret[high_variance_indices[i]]=high_variance_member_embedding[i][high_variance_members[i]]
        

        low_variance_mask = (variances <= threshold).float()
        low_variance_indices = low_variance_mask.nonzero(as_tuple=True)[0]
        low_variance_group_embedding=torch.sum(weight[low_variance_indices] * h[low_variance_indices], dim=1)
        for i in range(low_variance_indices.shape[0]):
            ret[low_variance_indices[i]] = low_variance_group_embedding[i]
        # soft----------------------------------------------------
        max_weightmemver_indices = torch.argmax(weight, dim=1)
        classification_scores = self.classifier(weight.squeeze(2))  # 
        classification_scores[:,1]=classification_scores[:,1]+self.bias
        predicted_classes = torch.argmax(torch.softmax(classification_scores, dim=1), dim=1)  # 
        ret=torch.zeros(x.shape[0],x.shape[2])
        for i in range(len(predicted_classes)):
            if predicted_classes[i] ==1:#leadership
                ret[i]=h[i][max_weightmemver_indices[i]]
            else:
                ret[i] = torch.sum(weight[i] * h[i], dim=0)
        return ret,weight,predicted_classes
