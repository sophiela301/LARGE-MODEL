import pickle
import copy
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
from collections import defaultdict


def dataload(args):

    group_item_file = f'./data/{args.dataname}/group_item.csv'
    GI_train, gi_data_pre = data_split(group_item_file, args.group_split_ratio)

    user_item_file = f'./data/{args.dataname}/user_item.csv'
    UI_train, ui_data_pre = data_split(user_item_file, args.user_split_ratio)

    GU = pd.read_csv(f'./data/{args.dataname}/group_user.csv', sep=',', header=None,
                     names=['group_id', 'user_id'])
    start_idx, end_idx = GU['group_id'].min(), GU['group_id'].max()
    n_groups = end_idx - start_idx + 1
    df_ug_train = GU[GU.group_id.isin(range(start_idx, end_idx + 1))]
        # sort in ascending order of group ids.
    df_ug_train = df_ug_train.sort_values('group_id')
    max_group_size = df_ug_train.groupby(  # 22
            'group_id').size().max()  # max group size denoted by G
    g_u_list_train = df_ug_train.groupby(
            'group_id')['user_id'].apply(list).reset_index()
    g_u_list_train['user_id'] = list(map(lambda x: x + [602] * (max_group_size - len(x)),  #camra
                                          g_u_list_train.user_id))
    data_gu = np.squeeze(
            np.array(g_u_list_train[['user_id']].values.tolist()))
# .....................................

    GU['group_id'] = GU['group_id'].astype('category')
    GU['user_id'] = GU['user_id'].astype('category')
    group_id = GU['group_id'].cat.codes.values
    user_id = GU['user_id'].cat.codes.values
    group_num, user_num = group_id.max() + 1, user_id.max() + 1
    link = np.ones(len(GU['group_id']))
    GU = csr_matrix((link, (group_id, user_id)), shape=(group_num, user_num))

    G_train, U_train = GU.nonzero()
    g_u_dict = defaultdict(list)
    for i in range(len(G_train)):
        g_u_dict[G_train[i]].append(U_train[i])
    u_max_i = max([len(i) for i in g_u_dict.values()])  
    for i, id_list in g_u_dict.items():  
        if len(id_list) < u_max_i:
            g_u_dict[i] += [602] * (u_max_i - len(id_list))
    u_id_train = np.array(list(set(G_train)), dtype=np.int32)
    g_u_list = []  
    for i in range(len(u_id_train)):
        g_u_list.append(g_u_dict[i])
    members = []
    gid = []
    for i in range(len(g_u_list)):
        if i in gi_data_pre['h_train']:
            members.append(g_u_list[i])
            gid.append(i)
    g_items_num = []
    member_items_num = []
    for i in range(len(members)):
        g_items_num.append(len(gi_data_pre['non_item_list'][i]))
        num = 0
        for id in members[i]:
            if id in ui_data_pre['h_train']:
                num += len(ui_data_pre['non_item_list']
                           [list(ui_data_pre['h_train']).index(id)])
        member_items_num.append(num)
   
    g_max = max(i for i in g_items_num)  # 
    g_min = min(i for i in g_items_num)  # 
    u_max = max(i for i in member_items_num)  # 
    u_min = min(i for i in member_items_num)  # 


    xiaoyu = []  
    bei = []
    shibei = []
    baibei = []
    # more =[]
     
    for i in gi_data_pre['h_train']:

        if ((member_items_num[list(gi_data_pre['h_train']).index(i)])/(g_items_num[list(gi_data_pre['h_train']).index(i)] )<=0.25):
            xiaoyu.append(i)
        if (0.25<(member_items_num[list(gi_data_pre['h_train']).index(i)])/(g_items_num[list(gi_data_pre['h_train']).index(i)] )<=0.5):
            bei.append(i)
        if (0.5<(member_items_num[list(gi_data_pre['h_train']).index(i)])/(g_items_num[list(gi_data_pre['h_train']).index(i)] )<=0.75):
            shibei.append(i)
        if (0.75<(member_items_num[list(gi_data_pre['h_train']).index(i)])/(g_items_num[list(gi_data_pre['h_train']).index(i)] )<=1):
            baibei.append(i)
       
        elif (100<(member_items_num[list(gi_data_pre['h_train']).index(i)])/(g_items_num[list(gi_data_pre['h_train']).index(i)] )):
            baibei.append(i)
    return GI_train, UI_train, GU, gi_data_pre, ui_data_pre,data_gu


def data_split(file, split_ratio):
    UI = pd.read_csv(file, sep=',', header=None,
                     names=['user_id', 'item_id'])
    UI['user_id'] = UI['user_id'].astype('category')
    UI['item_id'] = UI['item_id'].astype('category')
    user_id = UI['user_id'].cat.codes.values
    item_id = UI['item_id'].cat.codes.values
    user_num = user_id.max() + 1
    item_num = item_id.max() + 1

    link = np.ones(len(UI['user_id']))
    UI_train = csr_matrix((link, (user_id, item_id)),
                          shape=(user_num, item_num))
    UI_valid = csr_matrix((link, (user_id, item_id)),
                          shape=(user_num, item_num))
    UI_test = csr_matrix((link, (user_id, item_id)),
                         shape=(user_num, item_num))

    float_mask = np.random.permutation(np.linspace(0, 1, len(UI['user_id'])))
    UI_train.data[float_mask >= split_ratio['train']] = 0
    UI_valid.data[(float_mask < split_ratio['train']) | (
        float_mask > (1 - split_ratio['test']))] = 0
    UI_test.data[float_mask <= (1 - split_ratio['test'])] = 0

    link = np.ones(len(UI_train.nonzero()[0]))
    UI_train = csr_matrix((link, UI_train.nonzero()),
                          shape=(user_num, item_num))


    u_train, i_train = UI_train.nonzero()
    u_valid, i_valid = UI_valid.nonzero()
    u_test, i_test = UI_test.nonzero()

    n_head, n_tail = UI_train.shape
    u_iid_dict = defaultdict(list)
    for i in range(len(u_train)):
        u_iid_dict[u_train[i]].append(i_train[i])
    u_iid_dict1 = copy.deepcopy(u_iid_dict)
    u_iid_list_no = []  
    for i, id_list in u_iid_dict1.items(): 
        u_iid_list_no.append(u_iid_dict1[i])

    u_max_i = max([len(i) for i in u_iid_dict.values()])  
    # u_min_i = min([len(i) for i in u_iid_dict.values()])  # 50

   
    for i, id_list in u_iid_dict.items():
        if len(id_list) < u_max_i:
            u_iid_dict[i] += [item_num] * (u_max_i - len(id_list))

    u_id_train = np.array(list(set(u_train)), dtype=np.int32)
    u_iid_list = []
    for i in u_id_train:
        u_iid_list.append(u_iid_dict[i])

    train_data = np.column_stack((u_train, i_train))
    valid_data = np.column_stack((u_valid, i_valid))
    test_data = np.column_stack((u_test, i_test))

    u_valid = np.array(list(set(u_valid)), dtype='int64')
    u_test = np.array(list(set(u_test)), dtype='int64')

    valid_mask = torch.zeros(n_head, n_tail)

    for (u, i) in train_data:
        valid_mask[u][i] = -np.inf

    test_mask = valid_mask.clone()
    valid_ground_truth_list = [[] for _ in range(n_head)]
    for (u, i) in valid_data:
        valid_ground_truth_list[u].append(i)
        test_mask[u][i] = -np.inf

    test_ground_truth_list = [[] for _ in range(n_head)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    data = {
        'h_train': u_id_train,
        'h_valid': u_valid,
        'h_test': u_test,
        'h_iid_list': u_iid_list,
        'valid_mask': valid_mask,
        'test_mask': test_mask,
        'valid_ground_truth_list': valid_ground_truth_list,
        'test_ground_truth_list': test_ground_truth_list,
        'non_item_list': u_iid_list_no
    }

    return UI_train, data
