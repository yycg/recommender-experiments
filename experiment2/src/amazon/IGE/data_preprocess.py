import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import contextlib
import argparse
import gzip
import json
import itertools


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/sjy2018/Amazon/')
    args = parser.parse_known_args()[0]

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

        all_users = pd.DataFrame({'user_id': list(action_data['user_id'].unique())})
        user_lable = LabelEncoder()
        user_lable.fit(all_users['user_id'])
        action_data['user_id'] = user_lable.transform(action_data['user_id'])

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

    with elapsed_timer("-- {0}s - %s" % ("get induced edge list",)):
        user_items_map = {}
        item_users_map = {}
        for index, row in action_data_train.iterrows():
            user = row["user_id"]
            item = row["sku_id"]
            user_items_map.setdefault(user, [])
            user_items_map[user].append(item)
            item_users_map.setdefault(item, [])
            item_users_map[item].append(user)

        # s_u: (u-v-v)
        with open("../../../data/amazon/s_u.csv", "w") as file:
            for user, items in user_items_map.items():
                permutations = itertools.permutations(items, 2)
                for permutation in permutations:
                    file.write(str(user) + " " + " ".join([str(item) for item in permutation]) + "\n")
                for item in items:
                    file.write(str(user) + " " + " ".join([str(item), str(item)]) + "\n")

        # s_v: (v-u-u)
        with open("../../../data/amazon/s_v.csv", "w") as file:
            for item, users in item_users_map.items():
                permutations = itertools.permutations(users, 2)
                for permutation in permutations:
                    file.write(str(item) + " " + " ".join([str(user) for user in permutation]) + "\n")
                for user in users:
                    file.write(str(item) + " " + " ".join([str(user), str(user)]) + "\n")

    with elapsed_timer("-- {0}s - %s" % ("statistic",)):
        max_user_id = -1
        max_item_id = -1
        max_brand_id = -1

        for index, row in action_data.iterrows():
            user = row["user_id"]
            item = row["sku_id"]
            max_user_id = max(max_user_id, user)
            max_item_id = max(max_item_id, item)

        for index, row in sku_side_info.iterrows():
            brand = row["brand"]
            max_brand_id = max(max_brand_id, brand)

        print("#users: " + str(max_user_id+1))
        print("#items: " + str(max_item_id + 1))
        print("#edges: " + str(len(action_data)))
        print("#brands: " + str(max_brand_id + 1))

    with elapsed_timer("-- {0}s - %s" % ("get adjacency list",)):
        with open("../../../data/amazon/user_adjacency_list.csv", "w") as file:
            for user, items in user_items_map.items():
                for item in items:
                    file.write(str(user) + " " + str(item) + "\n")

        with open("../../../data/amazon/item_adjacency_list.csv", "w") as file:
            for item, users in item_users_map.items():
                for user in users:
                    file.write(str(item) + " " + str(user) + "\n")
