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

