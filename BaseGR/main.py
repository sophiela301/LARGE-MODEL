import os
from scipy.sparse import csr_matrix, hstack, vstack
import torch.nn as nn
import random
import dgl
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
import time

from data_pre import dataload
from model import *
from utils import *
from aggregators import AttentionAggregator

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def matrix_cat(inter_M):
    n_head, n_tail = inter_M.shape
    top_left = csr_matrix(np.eye(n_head, dtype=int))
    bottom_right = csr_matrix(np.eye(n_tail, dtype=int))
    top = hstack((top_left, inter_M))
    bottom = hstack((inter_M.T, bottom_right))
    A = vstack((top, bottom))
    return A


def collate_single(batch):
    gids, g_iid_list, uids, u_iid_list = zip(*batch)
    return torch.LongTensor(gids).to(args.device), torch.LongTensor(g_iid_list).to(args.device), torch.LongTensor(
        uids).to(args.device), torch.LongTensor(u_iid_list).to(args.device)


def single(graphs, gi_data_pre, ui_data_pre,data_gu):
    model = GROUP_single(args).to(args.device)
    preference_aggregator = AttentionAggregator(64, 64)
    print("model_single:")
    print(model)
    print('-----------------------------------------------')

    train_data = RecData(gi_data_pre['h_train'], gi_data_pre['h_iid_list'], ui_data_pre['h_train'],
                         ui_data_pre['h_iid_list'])
    train_dataloader = DataLoader(
        train_data, args.batch_size, shuffle=True, collate_fn=collate_single)

    opt1 = torch.optim.Adam(
        model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    best_recall = float('-inf')

    cnt_wait = 0
    epoch = 0
    while True:
        epoch += 1
        total_loss = 0.0

        model.train()
        opt1.zero_grad()
        gu = torch.tensor(data_gu)
        for _, (gids, g_iid_list, uids, u_iid_list,) in enumerate(train_dataloader):
            h_group, h_user, h_item = model(graphs)  
            group_embed_attention,weights,targets = preference_aggregator(member_embed, group_mask, args.variance,mlp=False)
            h_group = group_embed_attention.cuda(1)
# ................
            weight_loss=model.weight_loss(weights,targets)
            loss = model.loss(h_group, h_user, h_item, gids,
                              g_iid_list, uids, u_iid_list)
            opt1.zero_grad()
            loss=loss+1000*weight_loss
            total_loss += loss.item()
            loss.backward()
            opt1.step()
        print()

        print(f"{now()}\tEpoch: {epoch}\tcnt_wait:{cnt_wait}\tLoss = {total_loss}")

        model.eval()
        with torch.no_grad():
            gids = list(gi_data_pre['h_valid'])
            score = model.rank(model.h_group[gids], model.h_item)
            F1_score, Precision, Recall, NDCG = test(
                score, gids, gi_data_pre['valid_ground_truth_list'], gi_data_pre['valid_mask'], args.top_k)
        print("F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(
            F1_score, Precision, Recall, NDCG))
        with torch.no_grad():
            print('test:')
            gids = list(gi_data_pre['h_test'])
            score = model.rank(model.h_group[gids], model.h_item)
            tF1_score, tPrecision, tRecall, tNDCG = test(score, gids, gi_data_pre['test_ground_truth_list'],
                                                         gi_data_pre['test_mask'], args.top_k)
            F1_score_50, Precision_50, Recall_50, ndcg_50 = test(score, gids, gi_data_pre['test_ground_truth_list'],
                                                      gi_data_pre['test_mask'], 50)
        print(
            "F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(F1_score_50, Precision_50, Recall_50, ndcg_50))

        print(
            "F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(tF1_score, tPrecision, tRecall, tNDCG))

        if Recall > best_recall:
            best_recall = Recall
            cnt_wait = 0
            torch.save(model.state_dict(), f'model-{args.train_time}.pkl')

        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('#################################################################################################')
            print('Early stopping')
            break
    model.load_state_dict(torch.load(f'model-{args.train_time}.pkl'))
    model.eval()
    with torch.no_grad():
        print('test:')
        gids = list(gi_data_pre['h_test'])
        score = model.rank(model.h_group[gids], model.h_item)
        F1_score, Precision, Recall, NDCG = test_last(score, gids, gi_data_pre['test_ground_truth_list'],
                                                      gi_data_pre['test_mask'], args.top_k)
        F1_score_50, Precision_50, Recall_50, NDCG_50 = test_last(score, gids, gi_data_pre['test_ground_truth_list'],
                                                      gi_data_pre['test_mask'], 50)
    print('top20:')
    print(
        "F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(F1_score, Precision, Recall, NDCG))
    print('top50:')
    print(
        "F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(F1_score_50, Precision_50, Recall_50, NDCG_50))

def test(rating, uid, test_ground_truth_list, mask, topk):

    rating_list = []
    groundTrue_list = []

    rating = np.delete(rating.cpu(), -1, axis=1)
    rating += mask[uid]

    _, rating_K = torch.topk(rating, k=topk)
    rating_list.append(rating_K)

    groundTrue_list.append([test_ground_truth_list[u] for u in uid])

    X = zip(rating_list, groundTrue_list)
    # X = zip(rating_list[0], groundTrue_list[0])
    Recall, Precision, NDCG = 0, 0, 0

    NDCG_list = []
    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        NDCG_list.append(ndcg)
    # print(NDCG)
    # print(Recall)
    Precision /= len(uid)
    Recall /= len(uid)
    NDCG /= len(uid)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


def test_last(rating, uid, test_ground_truth_list, mask, topk):

    rating_list = []
    groundTrue_list = []

    rating = np.delete(rating.cpu(), -1, axis=1)
    rating += mask[uid]

    _, rating_K = torch.topk(rating, k=topk)
    rating_list.append(rating_K)

    groundTrue_list.append([test_ground_truth_list[u] for u in uid])

    X = zip(rating_list, groundTrue_list)
    # X = zip(rating_list[0], groundTrue_list[0])
    Recall, Precision, NDCG = 0, 0, 0

    NDCG_list = []
    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch_last(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        NDCG_list.append(ndcg)
    # print(NDCG)
    # print(Recall)
    Precision /= len(uid)
    Recall /= len(uid)
    NDCG /= len(uid)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == '__main__':
    args = parse_args_camara()
    param_grid = {
        'variance': [0.05]
    }
    print(f'param_grid:{param_grid}')
    args.train_time = str(time.strftime('%m-%d-%H-%M'))

    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2])
                            for x in open('tmp', 'r').readlines()]
        # gpu_id = int(np.argmax(memory_available))
        gpu_id =1
        args.device = 'cuda:{}'.format(gpu_id)
    else:
        args.device = 'cpu'

    for variance in param_grid['variance']:
        args.variance = variance
        # args.top_k = dropout
        setup_seed(2022)
        print(args)
        print('-----------------------------------------------')
        GI, UI, GU, gi_data_pre, ui_data_pre,data_gu = dataload(args)
        args.n_node = {'group': GI.shape[0],
                       'user': UI.shape[0],
                       'item': GI.shape[1]
                       }
        graph_GI, graph_GU, graph_UI = matrix_cat(
            GI), matrix_cat(GU), matrix_cat(UI)
        graphs = {'GI': dgl.DGLGraph(graph_GI).to(args.device),
                  'GU': dgl.DGLGraph(graph_GU).to(args.device),
                  'UI': dgl.DGLGraph(graph_UI).to(args.device)}

        eval(args.mode)(graphs, gi_data_pre, ui_data_pre,data_gu)
        print('#################################################################################################')
    print('test over !!!')
