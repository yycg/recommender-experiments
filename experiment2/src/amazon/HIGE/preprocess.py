import os
from gensim.models import KeyedVectors
import pandas as pd
import itertools
import time
from sklearn.preprocessing import LabelEncoder
import argparse
from datetime import datetime
import gzip
import json
import contextlib
from queue import PriorityQueue


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


def preprocess_net(args):
    # with elapsed_timer("-- {0}s - %s" % ("find topk timestamp",)):
    #     q = PriorityQueue()
    #     k = 500000
    #     with open(args.data_path + 'Clothing_Shoes_and_Jewelry.csv', 'r') as action:
    #         for line in action:
    #             action_columns = line.split(',')
    #             q.put(int(action_columns[3]))
    #             if q.qsize() > k:
    #                 q.get()
    #     top_k_timestamp = q.get()
    #     print("top_k_timestamp: " + str(top_k_timestamp))
    #
    # item_ids = set()
    # with elapsed_timer("-- {0}s - %s" % ("reduce amazon dataset size",)):
    #     with open(args.data_path + 'Clothing_Shoes_and_Jewelry.csv', 'r') as action:
    #         with open(args.data_path + 'Clothing_Shoes_and_Jewelry_reduced.csv', 'w') as action_reduced:
    #             for line in action:
    #                 action_columns = line.split(',')
    #                 if int(action_columns[3]) >= top_k_timestamp:
    #                     action_reduced.write(line)
    #                     item_ids.add(action_columns[0])
    #
    # with elapsed_timer("-- {0}s - %s" % ("reduce meta dataset size",)):
    #     with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry.json', 'r') as meta:
    #         with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json', 'w') as meta_reduced:
    #             for line in meta:
    #                 meta_map = json.loads(line)
    #                 if meta_map["asin"] in item_ids:
    #                     meta_reduced.write(line)
    #     with open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json', "rb") as meta_reduced:
    #         with gzip.open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json.gz', "wb") as meta_reduced_gz:
    #             meta_reduced_gz.writelines(meta_reduced)

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

        with open("../../../data/amazon/net.txt", "w") as net:
            for index, row in action_data_train.iterrows():
                user = row["user_id"]
                item = row["sku_id"]
                net.write(str(user) + " " + str(item) + " 1\n")

        user_set = set(action_data['user_id'].to_list())
        item_set = set(action_data['sku_id'].to_list())
        with open("../../../data/amazon/field.txt", "w") as field:
            for user in user_set:
                field.write(str(user) + " u\n")
            for item in item_set:
                field.write(str(item) + " i\n")

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

        category_column = [categories[-1] for categories in sku_category["category"].to_list()]
        sku_side_info["category"] = category_column
        sku_side_info.to_csv('../../../data/amazon/sku_side_info_category.csv', index=False, header=False, sep='\t')

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

        with open("../../../data/amazon/category_side_info.csv", "w", encoding="utf-8") as file:
            for i in range(trie.num_category):
                node = trie.index_category_map[i]
                # file.write(str(i) + "\t" + node.val + "\n")
                file.write(str(i) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='D:/Developer/Amazon/')
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--q", type=float, default=2)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_known_args()[0]

    preprocess_net(args)

