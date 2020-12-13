import pandas as pd
import numpy as np
from itertools import chain
import pickle
import time
import networkx as nx
from walker import RandomWalker
from sklearn.preprocessing import LabelEncoder
import argparse
from datetime import datetime
import gzip
import json


def cnt_session(data, time_cut=30, cut_type=2):
    sku_list = data['sku_id']
    time_list = data['action_time']
    type_list = data['type']
    session = []
    tmp_session = []
    for i, item in enumerate(sku_list):
        # if type_list[i] == cut_type or (i < len(sku_list)-1 and (time_list[i+1] - time_list[i]).seconds/60 > time_cut) or i == len(sku_list)-1:
        #     tmp_session.append(item)
        #     session.append(tmp_session)
        #     tmp_session = []
        # else:
        #     tmp_session.append(item)
        if i == len(sku_list)-1:
            tmp_session.append(item)
            session.append(tmp_session)
            tmp_session = []
        else:
            tmp_session.append(item)
    return session


def get_session(action_data, use_type=None):
    if use_type is None:
        use_type = [1, 2, 3, 5]
    action_data = action_data[action_data['type'].isin(use_type)]
    action_data = action_data.sort_values(by=['user_id', 'action_time'], ascending=True)
    group_action_data = action_data.groupby('user_id').agg(list)
    session_list = group_action_data.apply(cnt_session, axis=1)
    return session_list.to_numpy()


def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    return np.array(all_pairs, dtype=np.int32)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/sjy2018/Amazon/')
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--q", type=float, default=2)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_known_args()[0]

    action_data = pd.read_csv(args.data_path + 'Clothing_Shoes_and_Jewelry.csv', header=None,
                              names=["user_id", "sku_id", "rating", "timestamp"])
    action_data.sort_values("timestamp", inplace=True)
    timestamps = action_data["timestamp"].tolist()
    datetimes = []
    for timestamp in timestamps:
        datetimes.append(datetime.fromtimestamp(timestamp))
    action_data["action_time"] = datetimes
    action_data["type"] = 1

    # filter
    user_count = {}
    sku_count = {}
    for index, row in action_data.iterrows():
        user_id = row["user_id"]
        sku_id = row["sku_id"]
        user_count.setdefault(user_id, 0)
        user_count[user_id] += 1
        sku_count.setdefault(sku_id, 0)
        sku_count[sku_id] += 1

    index_list = []
    for index, row in action_data.iterrows():
        user_id = row["user_id"]
        sku_id = row["sku_id"]
        if 5 <= user_count[user_id] <= 20 and 5 <= sku_count[sku_id] <= 20:
            index_list.append(index)
    action_data = action_data.loc[index_list]

    # split
    test_ratio = 0.25

    user_items_map = {}
    for index, row in action_data.iterrows():
        user = row["user_id"]
        item = row["sku_id"]
        user_items_map.setdefault(user, [])
        user_items_map[user].append({"item": item, "index": index})

    train_index = []
    test_index = []
    for user, maps in user_items_map.items():
        for i in range(len(maps)):
            if i < (1 - test_ratio) * len(maps):
                train_index.append(maps[i]["index"])
            else:
                test_index.append(maps[i]["index"])

    action_data_test = action_data.loc[test_index]
    action_data = action_data.loc[train_index]

    with open("../../../data/amazon/user-event-rsvp_test.tsv", "w") as test:
        for index, row in action_data_test.iterrows():
            user = row["user_id"]
            item = row["sku_id"]
            test.write(str(user) + "\t" + str(item) + "\n")

    all_skus = action_data['sku_id'].unique()
    all_skus = pd.DataFrame({'sku_id': list(all_skus)})
    sku_lbe = LabelEncoder()
    all_skus['sku_id'] = sku_lbe.fit_transform(all_skus['sku_id'])
    action_data['sku_id'] = sku_lbe.transform(action_data['sku_id'])

    print('make session list\n')
    start_time = time.time()
    session_list = get_session(action_data, use_type=[1, 2, 3, 5])
    session_list_all = []
    for item_list in session_list:
        for session in item_list:
            if len(session) > 1:
                session_list_all.append(session)

    print('make session list done, time cost {0}'.format(str(time.time() - start_time)))

    # session2graph
    node_pair = dict()
    for session in session_list_all:
        for i in range(1, len(session)):
            if (session[i - 1], session[i]) not in node_pair.keys():
                node_pair[(session[i - 1], session[i])] = 1
            else:
                node_pair[(session[i - 1], session[i])] += 1

    in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
    out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
    weight_list = list(node_pair.values())
    graph_df = pd.DataFrame({'in_node': in_node_list, 'out_node': out_node_list, 'weight': weight_list})
    graph_df.to_csv('../../../data/amazon/graph.csv', sep=' ', index=False, header=False)

    G = nx.read_edgelist('../../../data/amazon/graph.csv', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    walker = RandomWalker(G, p=args.p, q=args.q)
    print("Preprocess transition probs...")
    walker.preprocess_transition_probs()

    session_reproduce = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, workers=4,
                                              verbose=1)
    session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    # add side info
    df = getDF(args.data_path + 'meta_Clothing_Shoes_and_Jewelry.json.gz')
    product_data = df.loc[:, ["asin", "brand"]]
    product_data.rename(columns={'asin': 'sku_id'})

    all_skus['sku_id'] = sku_lbe.inverse_transform(all_skus['sku_id'])
    print("sku nums: " + str(all_skus.count()))
    sku_side_info = pd.merge(all_skus, product_data, on='sku_id', how='left').fillna(0)

    # id2index
    for feat in sku_side_info.columns:
        if feat != 'sku_id':
            lbe = LabelEncoder()
            sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
        else:
            sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])

    sku_side_info = sku_side_info.sort_values(by=['sku_id'], ascending=True)
    sku_side_info.to_csv('../../../data/amazon/sku_side_info.csv', index=False, header=False, sep='\t')

    # get pair
    all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
    np.savetxt('../../../data/amazon/all_pairs', X=all_pairs, fmt="%d", delimiter=" ")
