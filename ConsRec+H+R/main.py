import torch
import random
import torch.optim as optim
import numpy as np
from metrics import evaluate
from model import ConsRec
from datetime import datetime
import argparse
import time
from dataloader import GroupDataset
# from tensorboardX import SummaryWriter
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
large_penalty = 100  
classifier_threshold = 0.5  
def weight_loss(weights, targets):  
        weights = weights.squeeze(-1)
        leader_indices = torch.where(targets == 1)[0]  
        collaboration_indices = torch.where(targets == 0)[0]
        if len(leader_indices) == 0: 
            return large_penalty
        # Calculate indicators for leader groups
        leader_weights = weights[leader_indices]
        leader_indicators = calculate_indicator(
            leader_weights, 1)  
        leader_indicators_repeated = leader_indicators.unsqueeze(
            1).expand(-1, 4)  

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(
            0, collaboration_indices.size(0), (leader_indices.size(0), 2))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = calculate_indicator(
            selected_collaboration_weights, 0)  

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated),
                            classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty
def calculate_indicator(weights, isleader): 
        rest_weights = weights.clone()
        # Replace 0 weights with nan
        rest_weights[weights == 0] = float('nan')
        if isleader == 1:
            max_index = torch.argmax(weights, dim=1)
            max_weight = weights[torch.arange(weights.size(0)), max_index]

            rest_weights = torch.cat(
                [weights[:, :max_index[0]], weights[:, max_index[0]+1:]], dim=1)

            # rest_mean = torch.mean(rest_weights, dim=1)
            weights_clone = rest_weights.clone()
            weights_clone = torch.where(weights_clone == 0, torch.tensor(
                float('nan')).cuda(1), weights_clone)
            mask = torch.isnan(weights_clone)
            rest_mean = torch.sum(weights_clone.masked_fill_(
                mask, 0.), dim=1) / mask.logical_not().sum(dim=1)
            indicator = (max_weight - rest_mean) / rest_mean
        else:
            # Find index of maximum weight within each group
            max_index = torch.argmax(weights, dim=2)
            max_weight = weights.gather(
                2, max_index.unsqueeze(2)).squeeze(2) 
            weights_clone = weights.clone()
            max_index_exp = max_index.unsqueeze(2)
            # Set the max weights and 0 values to nan
            weights_clone = weights_clone.scatter_(
                2, max_index_exp, float('nan'))
            weights_clone = torch.where(weights_clone == 0, torch.tensor(
                float('nan')).cuda(1), weights_clone)
            # Mask where nan values
            mask = torch.isnan(weights_clone)
            # Calculate mean value while ignoring nan
            rest_mean = torch.sum(weights_clone.masked_fill_(
                mask, 0.), dim=2) / mask.logical_not().sum(dim=2)

            indicator = (max_weight - rest_mean) / rest_mean
        return indicator
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def training(train_loader, epoch, type_m="group"):
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(
            running_device), pi_ni[:, 1].to(running_device)

        if type_m == 'user':
            pos_prediction = train_model(None, user_input, pos_items_input)
            neg_prediction = train_model(None, user_input, neg_items_input)
        else:
            # pos_prediction = train_model(user_input, None, pos_items_input)
            pos_prediction,weights,targets = train_model(user_input, None, pos_items_input) 
            neg_prediction,_,_ = train_model(user_input, None, neg_items_input)

        optimizer.zero_grad()
        if args.loss_type == "BPR":
            loss = torch.mean(torch.nn.functional.softplus(
                neg_prediction - pos_prediction))
        else:
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
        
        weightloss = weight_loss(weights, targets)
        loss=loss+weightloss 
        losses.append(loss)
        loss.backward()
        optimizer.step()

    print(
        f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str,
                        help="[Mafengwo, CAMERa2011]", default="M622")
    parser.add_argument("--device", type=str,
                        help="[cuda:0, ..., cpu]", default="cuda:1")

    parser.add_argument("--layers", type=int,
                        help="# HyperConv & OverlapConv layers", default=3)
    parser.add_argument("--emb_dim", type=int,
                        help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--topK", type=list, default=[20, 50])

    parser.add_argument("--epoch", type=int, default=100,
                        help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")

    args = parser.parse_args()
    set_seed(args.seed)

    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"), flush=True)
    print(args)



    running_device = torch.device(args.device)

    # Load dataset
    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(
        user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    print(f" #Users {num_users}, #Items {num_items}, #Groups {num_groups}\n")

    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)
    light_gcn_graph = dataset.light_gcn_graph.to(running_device)

    # Prepare model
    train_model = ConsRec(num_users, num_items, num_groups, args, user_hg, item_hg,
                          full_hg, overlap_graph, running_device, light_gcn_graph, dataset.num_group_net_items)
    train_model.to(running_device)

    for epoch_id in range(args.epoch):
        train_model.train()
        group_loss = training(dataset.get_group_dataloader(
            args.batch_size), epoch_id, "group")

        hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives, running_device,
                               args.topK, 'group')

        print(
            f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
       

    print()
    print('## Finishing Time:', datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
    print("Done!")
