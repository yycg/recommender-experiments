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
import contextlib
from queue import PriorityQueue
import itertools
from deepwalk import graph
import random


class StubLogger(object):
    def __getattr__(self, name):
        return self.log_print

    def log_print(self, msg, *args):
        print(msg % args)


LOGGER = StubLogger()
LOGGER.info("Hello %s!", "world")


@contextlib.contextmanager
def elapsed_timer(message):
    start_time = time.time()
    yield
    LOGGER.info(message.format(time.time() - start_time))


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


class TrieNode(object):
    def __init__(self, val, leaf, children, index):
        self.val, self.leaf, self.children, self.index = val, leaf, children, index


class Trie(object):
    def __init__(self):
        self.root = TrieNode("", False, {}, -1)
        self.num_category = 0
        self.index_category_map = {}

    def insert(self, leaf, path):
        category_path = []

        node = self.root
        for category in path:
            if category not in node.children:
                node.children[category] = TrieNode(category, False, {}, self.num_category)
                self.index_category_map[self.num_category] = node.children[category]
                self.num_category += 1
            category_path.append(node.children[category].index)
            node = node.children[category]
        node.children[leaf] = TrieNode(leaf, True, {}, -1)

        return category_path


def load_category_edgelist(file_, undirected=True):
    category_graph_map = {}
    with open(file_) as f:
        for l in f:
            c, x, y = l.strip().split()[:3]
            x = int(x)
            y = int(y)
            category_graph_map.setdefault(c, graph.Graph())
            G = category_graph_map[c]
            G[x].append(y)
            if undirected:
                G[y].append(x)

    for category, G in category_graph_map.items():
        G.make_consistent()

    return category_graph_map


def random_walk(category_graph_map, num_paths, path_length, alpha=0, rand=random.Random(0)):
    for category, G in category_graph_map.items():
        nodes = list(G.nodes())

        for cnt in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                walk = G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
                yield [category] + walk


def get_category_graph_context_all_pairs(category_category_children_walks, category_item_children_walks,
                                         window_size, num_items):
    all_pairs = []

    # category_category_children_walks
    for walk in category_category_children_walks:
        for i in range(len(walk)):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 1 or j >= len(walk):
                    continue
                else:
                    # (category, category)
                    all_pairs.append([num_items + int(walk[i]), num_items + int(walk[j])])

    # category_item_children_walks
    for walk in category_item_children_walks:
        for i in range(len(walk)):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 1 or j >= len(walk):
                    continue
                elif i == 0:
                    # (category, item)
                    all_pairs.append([num_items + int(walk[i]), int(walk[j])])
                else:
                    # (item, item)
                    all_pairs.append([int(walk[i]), int(walk[j])])

    return np.array(all_pairs, dtype=np.int32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/sjy2018/Amazon/')
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--q", type=float, default=2)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_known_args()[0]

    with elapsed_timer("-- {0}s - %s" % ("find topk timestamp",)):
        q = PriorityQueue()
        k = 500000
        with open(args.data_path + 'Clothing_Shoes_and_Jewelry.csv', 'r') as action:
            for line in action:
                action_columns = line.split(',')
                q.put(int(action_columns[3]))
                if q.qsize() > k:
                    q.get()
        top_k_timestamp = q.get()
        print("top_k_timestamp: " + str(top_k_timestamp))

    item_ids = set()
    with elapsed_timer("-- {0}s - %s" % ("reduce amazon dataset size",)):
        with open(args.data_path + 'Clothing_Shoes_and_Jewelry.csv', 'r') as action:
            with open(args.data_path + 'Clothing_Shoes_and_Jewelry_reduced.csv', 'w') as action_reduced:
                for line in action:
                    action_columns = line.split(',')
                    if int(action_columns[3]) >= top_k_timestamp:
                        action_reduced.write(line)
                        item_ids.add(action_columns[0])

    with elapsed_timer("-- {0}s - %s" % ("reduce meta dataset size",)):
        with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry.json', 'r') as meta:
            with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json', 'w') as meta_reduced:
                for line in meta:
                    meta_map = json.loads(line)
                    if meta_map["asin"] in item_ids:
                        meta_reduced.write(line)
        with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json', "rb") as meta_reduced:
            with gzip.open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json.gz', "wb") as meta_reduced_gz:
                meta_reduced_gz.writelines(meta_reduced)

    with elapsed_timer("-- {0}s - %s" % ("read action data",)):
        action_data = pd.read_csv(args.data_path + 'Clothing_Shoes_and_Jewelry_reduced.csv', header=None,
                                  names=["sku_id", "user_id", "rating", "timestamp"])
        # action_data = action_data.loc[action_data["timestamp"] > 1514736000]
        action_data.sort_values("timestamp", inplace=True)
        timestamps = action_data["timestamp"].tolist()
        datetimes = []
        for timestamp in timestamps:
            datetimes.append(datetime.fromtimestamp(timestamp))
        action_data["action_time"] = datetimes
        action_data["type"] = 1

    with elapsed_timer("-- {0}s - %s" % ("filter",)):
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

    with elapsed_timer("-- {0}s - %s" % ("split",)):
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

        all_skus = action_data['sku_id'].unique()
        all_skus = pd.DataFrame({'sku_id': list(all_skus)})
        sku_lbe = LabelEncoder()
        # Fit label encoder and return encoded labels.
        all_skus['sku_id'] = sku_lbe.fit_transform(all_skus['sku_id'])
        # Transform labels to normalized encoding.
        action_data['sku_id'] = sku_lbe.transform(action_data['sku_id'])

        action_data_test = action_data.loc[test_index]
        action_data_train = action_data.loc[train_index]

        with open("../../../data/amazon/user-event-rsvp_test.tsv", "w") as test:
            for index, row in action_data_test.iterrows():
                user = row["user_id"]
                item = row["sku_id"]
                test.write(str(user) + "\t" + str(item) + "\n")

        with open("../../../data/amazon/train.tsv", "w") as test:
            for index, row in action_data_train.iterrows():
                user = row["user_id"]
                item = row["sku_id"]
                test.write(str(user) + "\t" + str(item) + "\n")

    with elapsed_timer("-- {0}s - %s" % ("make session list",)):
        print('make session list\n')
        start_time = time.time()
        session_list = get_session(action_data_train, use_type=[1, 2, 3, 5])
        session_list_all = []
        for item_list in session_list:
            for session in item_list:
                if len(session) > 1:
                    session_list_all.append(session)

        print('make session list done, time cost {0}'.format(str(time.time() - start_time)))

    with elapsed_timer("-- {0}s - %s" % ("session2graph",)):
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

    with elapsed_timer("-- {0}s - %s" % ("random walk",)):
        G = nx.read_edgelist('../../../data/amazon/graph.csv', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
        walker = RandomWalker(G, p=args.p, q=args.q)
        print("Preprocess transition probs...")
        walker.preprocess_transition_probs()

        session_reproduce = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, workers=4,
                                                  verbose=1)
        session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    with elapsed_timer("-- {0}s - %s" % ("add side info",)):
        df = getDF(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json.gz')
        product_data = df.loc[:, ["asin", "brand"]]
        product_data = product_data.rename(columns={'asin': 'sku_id'})

        # Transform labels back to original encoding.
        all_skus['sku_id'] = sku_lbe.inverse_transform(all_skus['sku_id'])
        print("sku nums: " + str(all_skus.count()))
        sku_side_info = pd.merge(all_skus, product_data, on='sku_id', how='left').fillna("NaN")

        # id2index
        for feat in sku_side_info.columns:
            if feat != 'sku_id':
                lbe = LabelEncoder()
                sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
            else:
                sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])

        sku_side_info = sku_side_info.sort_values(by=['sku_id'], ascending=True)
        sku_side_info.to_csv('../../../data/amazon/sku_side_info.csv', index=False, header=False, sep='\t')

    with elapsed_timer("-- {0}s - %s" % ("get pair",)):
        all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
        np.savetxt('../../../data/amazon/all_pairs', X=all_pairs, fmt="%d", delimiter=" ")

    with elapsed_timer("-- {0}s - %s" % ("add category",)):
        product_data = df.loc[:, ["asin", "category"]]
        product_data = product_data.rename(columns={'asin': 'sku_id'})

        sku_category = pd.merge(all_skus, product_data, on='sku_id', how='left').fillna("NaN")

        sku_category['sku_id'] = sku_lbe.transform(sku_category['sku_id'])
        sku_category = sku_category.sort_values(by=['sku_id'], ascending=True)

        category_column = []
        trie = Trie()
        for index, row in sku_category.iterrows():
            leaf = row['sku_id']
            path = row['category']
            category_column.append(trie.insert(leaf, path))
        sku_category['category'] = category_column
        sku_category.to_csv('../../../data/amazon/sku_category.csv', index=False, header=False, sep='\t')

        category_item_children_map = {}
        category_category_children_map = {}
        for i in range(trie.num_category):
            node = trie.index_category_map[i]
            item_children = []
            category_children = []
            for _, child_node in node.children.items():
                if child_node.leaf:
                    item_children.append(child_node.val)
                else:
                    category_children.append(child_node.index)
            if len(item_children) > 0:
                category_item_children_map[i] = item_children
            if len(category_children) > 0:
                category_category_children_map[i] = category_children

        with open("../../../data/amazon/category_category_children.csv", "w") as file:
            for category, category_children in category_category_children_map.items():
                file.write(str(category) + "\t" + ",".join([str(category) for category in category_children]) + "\n")

        with open("../../../data/amazon/category_item_children.csv", "w") as file:
            for category, item_children in category_item_children_map.items():
                file.write(str(category) + "\t" + ",".join([str(item) for item in item_children]) + "\n")

    with elapsed_timer("-- {0}s - %s" % ("build graph",)):
        with open("../../../data/amazon/category_category_children.csv", "r") as reader, \
                open("../../../data/amazon/category_category_children_edge.txt", "w") as writer:
            for line in reader:
                columns = line.strip().split("\t")
                category = columns[0]
                items = columns[1].split(",")
                edges = itertools.permutations(items, 2)
                for edge in edges:
                    writer.write(category + " " + " ".join(edge) + "\n")

        with open("../../../data/amazon/category_item_children.csv", "r") as reader, \
                open("../../../data/amazon/category_item_children_edge.txt", "w") as writer:
            for line in reader:
                columns = line.strip().split("\t")
                category = columns[0]
                items = columns[1].split(",")
                edges = itertools.permutations(items, 2)
                for edge in edges:
                    writer.write(category + " " + " ".join(edge) + "\n")

    with elapsed_timer("-- {0}s - %s" % ("random walk",)):
        category_category_children_graph_map = load_category_edgelist("../../../data/amazon/category_category_children_edge.txt")
        category_category_children_walks = random_walk(category_category_children_graph_map, args.num_walks, args.walk_length)

        category_item_children_graph_map = load_category_edgelist("../../../data/amazon/category_item_children_edge.txt")
        category_item_children_walks = random_walk(category_item_children_graph_map, args.num_walks, args.walk_length)

    with elapsed_timer("-- {0}s - %s" % ("get pair",)):
        num_items = len(all_skus['sku_id'])
        category_all_pairs = get_category_graph_context_all_pairs(category_category_children_walks,
                                                                  category_item_children_walks,
                                                                  args.window_size, num_items)
        np.savetxt('../../../data/amazon/category_all_pairs', X=category_all_pairs, fmt="%d", delimiter=" ")
