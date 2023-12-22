from torch.utils.data import Dataset
import numpy as np
import argparse


def parse_args_camara():
    print('-----------------------------------------------')
    parser = argparse.ArgumentParser(description='my_model')
    parser.add_argument('--dataname', type=str,
                        default='camra', help='Name of dataset.')
    parser.add_argument("--hid_dim", type=dict, default=64,
                        help='Hidden layer dim.')
    parser.add_argument("--batch_size", type=dict,
                        default=512, help='batch_size')  
    parser.add_argument('--lr1', type=float, default=0.001,
                        help='Learning rate of mvgrl.')
    parser.add_argument('--lr2', type=float, default=0.001,
                        help='Learning rate of linear evaluator.')
    parser.add_argument('--wd1', type=float, default=0.,
                        help='Weight decay of mvgrl.')
    parser.add_argument('--wd2', type=float, default=0.,
                        help='Weight decay of linear evaluator.')
    parser.add_argument('--neg_weight', type=dict,
                        default={'group': 2.5, 'user': 1.6}, help='neg_weight') 
    parser.add_argument("--drop_out", type=int, default=0.6,
                        help='drop_out.')  
    parser.add_argument('--top_k', type=int, default=20, help='top_k')
    parser.add_argument('--patience', type=int, default=40,
                        help='Patient epochs to wait before early stopping.')
    parser.add_argument('--mode', type=str, default='single',
                        help='single,dual,contrast')
    parser.add_argument('--layers', type=int, default=3, help='feature')
    parser.add_argument('--group_split_ratio', type=dict,
                        default={'train': 0.6, 'test': 0.2}, help='split_ratio')
    parser.add_argument('--user_split_ratio', type=dict,
                        default={'train': 0.6, 'test': 0.2}, help='split_ratio')
    parser.add_argument('--variance', type=float,
                        default=0.1, help='Threshold for determining whether to use attention')

    return parser.parse_args()


class RecData(Dataset):

    def __init__(self, gids, g_iid_dict, uids, u_iid_dict):
        self.gdata = list(zip(gids, g_iid_dict))
        self.udata = list(zip(uids, u_iid_dict))

    def __getitem__(self, gidx):
        assert gidx < len(self.gdata)
        uidx = np.random.randint(len(self.udata))
        return [*self.gdata[gidx], *self.udata[uidx]]

    def __len__(self):
        return len(self.gdata)


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)  
    precis_n = k
    recall_n = np.array([len(test_data)])

    recall_n = np.array([len(test_data[i])
                        for i in range(len(test_data))])  
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def RecallPrecision_ATk_last(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)  
    precis_n = k
    recall_n = np.array([len(test_data)])

    recall_n = np.array([len(test_data[i])
                        for i in range(len(test_data))]) 
    recall_n = np.where(recall_n != 0, recall_n, 1)
    # recall_list = right_pred / recall_n
    # f = open("anay-cmr-u", "w")
    # for line in list(recall_list):
    #     f.write(str(line)+'\n')
    # f.close()
    # print(recall_list)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))  # [1,20]
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def NDCGatK_r_last(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))  # [1,20]
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)

    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)


def test_one_batch_last(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk_last(groundTrue, r, k)

    return ret['precision'], ret['recall'], NDCGatK_r_last(groundTrue, r, k)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # groundTrue = test_data
        # predictTopK = pred_data

        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
