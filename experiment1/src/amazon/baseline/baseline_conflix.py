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
            with gzip.open(args.data_path + 'meta_Clothing_Shoes_and_Jewelry_reduced.json.gz',
                           "wb") as meta_reduced_gz:
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


def build_graph(data_path, user_items_train_map):
    with open(os.path.join(data_path, "user_item_edge.txt"), "w") as writer:
        for user, items in user_items_train_map.items():
            edges = itertools.permutations(items, 2)
            for edge in edges:
                writer.write(" ".join(edge) + " 1\n")


def get_user_cand_set(data_path):
    user_set = set()
    cand_set = set()
    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "r") as test_file:
        for line in test_file:
            line_list = line.strip().split("\t")
            user = line_list[0]
            cand = line_list[1]
            user_set.add(user)
            cand_set.add(cand)
    return user_set, cand_set


def get_user_items_train_map(data_path):
    user_items_train_map = {}
    with open(os.path.join(data_path, 'train.tsv'), 'r') as train:
        for line in train:
            columns = line.strip().split('\t')
            user = columns[0]
            item = columns[1]
            user_items_train_map.setdefault(user, [])
            user_items_train_map[user].append(item)

    return user_items_train_map

def run_model(data_path, smore_path, item2vec_path):
    # deepwalk
    cmd = smore_path + "/cli/deepwalk -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_dw.txt -undirected 1 -dimensions 64 -walk_times 10 -walk_steps 40 -window_size 5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # walklets
    cmd = smore_path + "/cli/walklets -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_wl.txt -undirected 1 -dimensions 64 -walk_times 10 -walk_steps 40 -window_min 2 " \
          "-window_max 5 -negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # line order=1
    cmd = smore_path + "/cli/line -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_line1.txt -undirected 1 -order 1 -dimensions 64 -sample_times 10 -negative_samples 5 " \
          "-alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # line order=2
    cmd = smore_path + "/cli/line -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_line2.txt -undirected 1 -order 2 -dimensions 64 -sample_times 10 -negative_samples 5 " \
          "-alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # hpe
    cmd = smore_path + "/cli/hpe -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_hpe.txt -undirected 1 -dimensions 64 -reg 0.01 -sample_times 5 -walk_steps 5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # app
    cmd = smore_path + "/cli/app -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_app.txt -undirected 1 -dimensions 64 -walk_times 100 -sample_times 20 -jump 0.5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # mf
    cmd = smore_path + "/cli/mf -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_mf.txt -dimensions 64 -sample_times 10 -negative_samples 5 -alpha 0.025 -reg 0.01 -threads 1"
    print(cmd)
    os.system(cmd)

    # bpr
    cmd = smore_path + "/cli/bpr -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_bpr.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # warp
    cmd = smore_path + "/cli/warp -train " + data_path + "/user_item_edge.txt -save " + data_path + \
          "/rep_warp.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # item2vec
    cmd = item2vec_path + "/fastText-0.9.1/fasttext skipgram -input " + data_path + "/user_item_list.txt -output " + \
          data_path + "/item2vec -minCount 0 -epoch 50 -neg 100"
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set, data_path, user_items_train_map):
    # deepwalk
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_dw.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "deepwalk.tsv")

    # walklets
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_wl.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "walklets.tsv")

    # line order=1
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line1.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "line1.tsv")

    # line order=2
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line2.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "line2.tsv")

    # hpe
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hpe.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "hpe.tsv")

    # app
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_app.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "app.tsv")

    # mf
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_mf.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "mf.tsv")

    # bpr
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_bpr.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "bpr.tsv")

    # warp
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_warp.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, "warp.tsv")

    # item2vec
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "item2vec.vec"), binary=False)
    _recommend_item2vec(user_set, cand_set, word_vectors, user_items_train_map, data_path)


def _recommend(user_set, cand_set, word_vectors, user_items_train_map, data_path, recommend_file):
    with open(os.path.join(data_path, recommend_file), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join(
                    [str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


def _recommend_item2vec(user_set, cand_set, word_vectors, user_items_train_map, data_path):
    with open(os.path.join(data_path, "item2vec.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item))
                             for item in user_items_train_map[user] if str(item) in word_vectors]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join(
                    [str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/sjy2018/Amazon/')
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--q", type=float, default=2)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_known_args()[0]

    preprocess_net(args)

    result_path = "../../../data/amazon"
    smore_path = "../../../../smore"
    item2vec_path = "../../../../item2vec"

    user_set, cand_set = get_user_cand_set(result_path)
    user_items_train_map = get_user_items_train_map(result_path)
    build_graph(result_path, user_items_train_map)
    run_model(result_path, smore_path, item2vec_path)
    recommend(user_set, cand_set, result_path, user_items_train_map)
