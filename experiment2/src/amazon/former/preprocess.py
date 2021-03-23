import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

amazon_path = "../../Amazon/AMAZON_FASHION.csv"
test_ratio = 0.25

CSE_path = "/media/shan/新加卷/Developer/smore"
data_path = "../../"
sample_times = 40
walk_steps = 5
alpha = 0.01


def preprocess_net():
    amazon = pd.read_csv(amazon_path, header=None, names=["user", "item", "rating", "timestamp"])

    user_set = set()
    item_set = set()
    for index, row in amazon.iterrows():
        user = row["user"]
        item = row["item"]
        user_set.add(user)
        item_set.add(item)

    user_item_rating_timestamp_map = {}
    for index, row in amazon.iterrows():
        user = row["user"]
        item = row["item"]
        rating = row["rating"]
        timestamp = row["timestamp"]
        if user in user_item_rating_timestamp_map:
            user_item_rating_timestamp_map[user].append((item, rating, timestamp, index))
        else:
            user_item_rating_timestamp_map[user] = [(item, rating, timestamp, index)]
    for user in user_item_rating_timestamp_map:
        user_item_rating_timestamp_map[user].sort(key=lambda tuple: tuple[2])

    train_index = []
    test_index = []
    test_user_set = set()
    test_cand_set = set()
    for user in user_item_rating_timestamp_map:
        for i in range(len(user_item_rating_timestamp_map[user])):
            item, rating, timestamp, index = user_item_rating_timestamp_map[user][i]
            if i < (1 - test_ratio) * len(user_item_rating_timestamp_map[user]):
                train_index.append(index)
            else:
                test_index.append(index)
                test_user_set.add(user)
                test_cand_set.add(item)

    amazon_train = amazon.iloc[train_index]
    amazon_test = amazon.iloc[test_index]

    with open(os.path.join(data_path, "net.txt"), "w") as net:
        for index, row in amazon_train.iterrows():
            user = row["user"]
            item = row["item"]
            rating = int(row["rating"])
            net.write(user + " " + item + " " + str(rating) + "\n")

    with open(os.path.join(data_path, "field.txt"), "w") as field:
        for user in user_set:
            field.write(str(user) + " u\n")
        for item in item_set:
            field.write(str(item) + " i\n")

    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "w") as test:
        for index, row in amazon_test.iterrows():
            user = row["user"]
            item = row["item"]
            test.write(user + "\t" + item + "\n")

    return test_user_set, test_cand_set


def run_model():
    # nemf
    cmd = CSE_path + "/cli/nemf -train " + data_path + "/net.txt -save " + data_path + "/rep_nemf.txt -field " \
          + data_path + "/field.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1" \
              .format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # nerank
    cmd = CSE_path + "/cli/nerank -train " + data_path + "/net.txt -save " + data_path + "/rep_nerank.txt -field " \
          + data_path + "/field.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1" \
              .format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # deepwalk
    cmd = CSE_path + "/cli/deepwalk -train " + data_path + "/net.txt -save " + data_path + \
          "rep_dw.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # walklets
    cmd = CSE_path + "/cli/walklets -train " + data_path + "/net.txt -save " + data_path + \
          "rep_wl.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # line order=1
    cmd = CSE_path + "/cli/line -train " + data_path + "/net.txt -save " + data_path + \
          "rep_line1.txt -order 1 -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # line order=2
    cmd = CSE_path + "/cli/line -train " + data_path + "/net.txt -save " + data_path + \
          "rep_line2.txt -order 2 -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # hpe
    cmd = CSE_path + "/cli/hpe -train " + data_path + "/net.txt -save " + data_path + \
          "rep_hpe.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # app
    cmd = CSE_path + "/cli/app -train " + data_path + "/net.txt -save " + data_path + \
          "rep_app.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # mf
    cmd = CSE_path + "/cli/mf -train " + data_path + "/net.txt -save " + data_path + \
          "rep_mf.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # bpr
    cmd = CSE_path + "/cli/bpr -train " + data_path + "/net.txt -save " + data_path + \
          "rep_bpr.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # warp
    cmd = CSE_path + "/cli/warp -train " + data_path + "/net.txt -save " + data_path + \
          "rep_warp.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)

    # hoprec
    cmd = CSE_path + "/cli/hoprec -train " + data_path + "/net.txt -save " + data_path + \
          "rep_hoprec.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1". \
              format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set):
    # nemf
    # load word vectors file
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_nemf.txt"), binary=False)

    with open(os.path.join(data_path, "nemf.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # nerank
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_nerank.txt"), binary=False)

    with open(os.path.join(data_path, "nerank.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # deepwalk
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_dw.txt"), binary=False)

    with open(os.path.join(data_path, "deepwalk.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # walklets
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_wl.txt"), binary=False)

    with open(os.path.join(data_path, "walklets.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # line order=1
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line1.txt"), binary=False)

    with open(os.path.join(data_path, "line1.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # line order=2
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line2.txt"), binary=False)

    with open(os.path.join(data_path, "line2.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # hpe
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hpe.txt"), binary=False)

    with open(os.path.join(data_path, "hpe.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # app
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_app.txt"), binary=False)

    with open(os.path.join(data_path, "app.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # mf
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_mf.txt"), binary=False)

    with open(os.path.join(data_path, "mf.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # bpr
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_bpr.txt"), binary=False)

    with open(os.path.join(data_path, "bpf.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # warp
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_warp.txt"), binary=False)

    with open(os.path.join(data_path, "warp.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # hoprec
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hoprec.txt"), binary=False)

    with open(os.path.join(data_path, "hoprec.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


if __name__ == "__main__":
    user_set, cand_set = preprocess_net()
    # run_model()
    recommend(user_set, cand_set)
