import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq
from collections import Counter
import torch.utils.data as data

class Helper(object):
    """
        utils class: it can provide any function that we need
    """

    def __init__(self):
        self.timber = True

    def evaluate_model(self, model, testRatings, testNegatives, device, K_list, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        user_test = []
        item_test = []
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = [rating[1]]
            items.extend(testNegatives[idx])
            item_test.append(items)
            user = np.full(len(items), rating[0])
            user_test.append(user)
        users_var = torch.LongTensor(user_test).to(device)
        items_var = torch.LongTensor(item_test).to(device)

        bsz = len(testRatings)
        item_len = len(testNegatives[0]) + 1

        users_var = users_var.view(-1)
        items_var = items_var.view(-1)
        if type_m == 'group':
            predictions,_,_ = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions,_,_ = model(None, users_var, items_var)
        predictions = torch.reshape(predictions, (bsz, item_len))
        pred_score = predictions.data.cpu().numpy()
        pred_rank = np.argsort(pred_score * -1, axis=1)
        for k in K_list:
            hits.append(getHitK(pred_rank, k))
            ndcgs.append(getNdcgK(pred_rank, k))
        return (hits, ndcgs)

    def valid_model(self, model, testRatings, dataset, device, K_list):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        user_test = []
        item_test = []
        neg_candidates = np.arange(dataset.num_items)

        idxs = np.random.choice(len(testRatings), len(testRatings))
        testNegatives = np.random.choice(neg_candidates, (len(testRatings), 99), replace=True)
        # item_test = np.repeat(np.arange(dataset.num_items).reshape(1, -1), dataset.num_groups, 0).flatten().tolist()
        i = 0
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = [rating[1]]
            items.extend(testNegatives[i])
            item_test.append(items)
            user = np.full(len(items), rating[0])
            user_test.append(user)
            i+=1
        users_var = torch.LongTensor(user_test).to(device)
        items_var = torch.LongTensor(item_test).to(device)

        bsz = len(testRatings)
        item_len = len(testNegatives[0]) + 1

        users_var = users_var.view(-1)
        items_var = items_var.view(-1)
        predictions,_,_ = model(users_var, None, items_var)
        predictions = torch.reshape(predictions, (bsz, item_len))
        pred_score = predictions.data.cpu().numpy()
        pred_rank = np.argsort(pred_score * -1, axis=1)
        hits.append(getHitK(pred_rank, 10))
        ndcgs.append(getNdcgK(pred_rank, 10))
        return (hits, ndcgs)

    def evaluate_model_group(self, model, dataset, device, K_list):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        f1, precision, recall, ndcg, entropy, item_coverage, gini_index = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        test_loader = data.DataLoader(list(np.arange(dataset.num_groups)), batch_size=100, shuffle=False)
        pred_scores = []
        for batch in test_loader:
            user_test = np.repeat(batch, dataset.num_items).tolist()
            item_test = np.repeat(np.arange(dataset.num_items).reshape(1, -1), len(batch), 0).flatten().tolist()
            users_var = torch.LongTensor(user_test).to(device)
            items_var = torch.LongTensor(item_test).to(device)

            users_var = users_var.view(-1)
            items_var = items_var.view(-1)
            predictions,_,_ = model(users_var, None, items_var)
            predictions = torch.reshape(predictions, (len(batch), dataset.num_items))
            pred_score = predictions.data.cpu()
            pred_score += dataset.test_mask[batch]
            pred_scores.append(pred_score)
        pred_scores = torch.vstack(pred_scores)
        for k in K_list:
            _, rating_list_all = torch.topk(pred_scores, k=k)
            rat = rating_list_all.numpy()
            entropy[k] = get_entropy(rat)
            item_coverage[k] = get_coverage(rat, dataset.num_items)
            gini_index[k] = get_gini(rat, dataset.num_items)

        # user_test = np.repeat(dataset.u_test, dataset.num_items).tolist()
        # item_test = np.repeat(np.arange(dataset.num_items).reshape(1, -1), len(dataset.u_test), 0).flatten().tolist()
        # users_var = torch.LongTensor(user_test).to(device)
        # items_var = torch.LongTensor(item_test).to(device)
        # users_var = users_var.view(-1)
        # items_var = items_var.view(-1)
        # predictions = model(users_var, None, items_var)
        # predictions = torch.reshape(predictions, (dataset.num_groups, dataset.num_items))
        # pred_score = predictions.data.cpu()
        # pred_score += dataset.test_mask[dataset.u_test]
        pred_scores = pred_scores[dataset.u_test]
        for k in K_list:
            _, rating_list_all = torch.topk(pred_scores, k=k)
            rating_list = []
            groundTrue_list = []

            rating_list.append(rating_list_all)
            groundTrue_list.append([dataset.test_ground_truth_list[u] for u in dataset.u_test])

            X = zip(rating_list, groundTrue_list)
            Recall, Precision, NDCG, Item_Coverage, Gini_Index = 0, 0, 0, 0, 0

            for i, x in enumerate(X):
                p, r, n = test_one_batch(x, k)
                Recall += r
                Precision += p
                NDCG += n
            precision[k] = Precision / len(dataset.u_test)
            recall[k] = Recall / len(dataset.u_test)
            ndcg[k] = NDCG / len(dataset.u_test)
            f1[k] = 2 * (precision[k] * recall[k]) / (precision[k] + recall[k])

            print(
                "F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}\tEntropy: {:.5f}\t Item_Coverage: {:.5f}\t Gini_Index: {:.5f}".format(
                    f1[k], precision[k], recall[k], ndcg[k], entropy[k], item_coverage[k], gini_index[k]))


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)

    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
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


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
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


def get_entropy(item_matrix):
    """Get shannon entropy through the top-k recommendation list.

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.

    Returns:
        float: the shannon entropy.
    """

    item_count = dict(Counter(item_matrix.flatten()))
    total_num = item_matrix.shape[0] * item_matrix.shape[1]
    result = 0.0
    for cnt in item_count.values():
        p = cnt / total_num
        result += -p * np.log(p)
    return result  # / len(item_count)


def get_coverage(item_matrix, num_items):
    """Get the coverage of recommended items over all items

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.
        num_items(int): the total number of items.

    Returns:
        float: the `coverage` metric.
    """
    unique_count = np.unique(item_matrix).shape[0]
    return unique_count / num_items


def get_gini(item_matrix, num_items):
    """Get gini index through the top-k recommendation list.

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.
        num_items(int): the total number of items.

    Returns:
        float: the gini index.
    """
    item_count = dict(Counter(item_matrix.flatten()))
    sorted_count = np.array(sorted(item_count.values()))
    num_recommended_items = sorted_count.shape[0]
    total_num = item_matrix.shape[0] * item_matrix.shape[1]
    idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
    gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
    gini_index /= num_items
    return gini_index


def getHitK(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return hit


def getNdcgK(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    ndcg = np.mean(ndcgs)
    return ndcg
