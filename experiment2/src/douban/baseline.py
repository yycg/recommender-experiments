import os
import pandas as pd
from sqlalchemy import create_engine
from gensim.models import KeyedVectors


def preprocess_net(data_path, test_ratio):
    print("Fetch data")
    engine = create_engine(
        "mysql+pymysql://douban_readonly:douban_readonly@10.105.240.25:3306/douban_beijing_2018?charset=utf8")
    sql = "select * from eventuser"
    eventuser = pd.read_sql_query(sql, engine)
    eventuser = eventuser.loc[eventuser["user_type"] == "participant"]

    sql = "select id from event"
    event_df = pd.read_sql_query(sql, engine)

    print("Preprocess data")
    event_set = set()
    for index, row in event_df.iterrows():
        event = row["id"]
        event_set.add(event)

    user_occurrence_count_map = {}
    item_occurrence_count_map = {}
    for index, row in eventuser.iterrows():
        user = row["user_id"]
        item = row["event_id"]
        user_occurrence_count_map.setdefault(user, 0)
        user_occurrence_count_map[user] += 1
        item_occurrence_count_map.setdefault(item, 0)
        item_occurrence_count_map[item] += 1

    user_set = set()
    item_set = set()
    user_items_map = {}
    for index, row in eventuser.iterrows():
        user = row["user_id"]
        item = row["event_id"]
        if 5 <= user_occurrence_count_map[user] < 20 and 5 <= item_occurrence_count_map[item] < 20 and item in event_set:
            user_set.add(user)
            item_set.add(item)
            user_items_map.setdefault(user, [])
            user_items_map[user].append({"item": item, "index": index})

    train_index = []
    test_index = []
    test_user_set = set()
    test_cand_set = set()
    for user, maps in user_items_map.items():
        for i in range(len(maps)):
            if i < (1 - test_ratio) * len(maps):
                train_index.append(maps[i]["index"])
            else:
                test_index.append(maps[i]["index"])
                test_user_set.add(user)
                test_cand_set.add(maps[i]["item"])

    eventuser_train = eventuser.loc[train_index]
    eventuser_test = eventuser.loc[test_index]

    print("Write data to files")
    with open(os.path.join(data_path, "net.txt"), "w") as net:
        for index, row in eventuser_train.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            net.write(str(user) + " " + str(item) + " 1\n")

    with open(os.path.join(data_path, "field.txt"), "w") as field:
        for user in user_set:
            field.write(str(user) + " u\n")
        for item in item_set:
            field.write(str(item) + " i\n")

    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "w") as test:
        for index, row in eventuser_test.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            test.write(str(user) + "\t" + str(item) + "\n")

    return test_user_set, test_cand_set


def run_model(smore_path, data_path):
    # nemf
    cmd = smore_path + "/cli/nemf -train " + data_path + "/net.txt -save " + data_path + "/rep_nemf.txt -field " \
          + data_path + "/field.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # nerank
    cmd = smore_path + "/cli/nerank -train " + data_path + "/net.txt -save " + data_path + "/rep_nerank.txt -field " \
          + data_path + "/field.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # deepwalk
    cmd = smore_path + "/cli/deepwalk -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_dw.txt -undirected 1 -dimensions 64 -walk_times 10 -walk_steps 40 -window_size 5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # walklets
    cmd = smore_path + "/cli/walklets -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_wl.txt -undirected 1 -dimensions 64 -walk_times 10 -walk_steps 40 -window_min 2 " \
          "-window_max 5 -negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # line order=1
    cmd = smore_path + "/cli/line -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_line1.txt -undirected 1 -order 1 -dimensions 64 -sample_times 10 -negative_samples 5 " \
          "-alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # line order=2
    cmd = smore_path + "/cli/line -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_line2.txt -undirected 1 -order 2 -dimensions 64 -sample_times 10 -negative_samples 5 " \
          "-alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # hpe
    cmd = smore_path + "/cli/hpe -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_hpe.txt -undirected 1 -dimensions 64 -reg 0.01 -sample_times 5 -walk_steps 5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # app
    cmd = smore_path + "/cli/app -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_app.txt -undirected 1 -dimensions 64 -walk_times 100 -sample_times 20 -jump 0.5 " \
          "-negative_samples 5 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # mf
    cmd = smore_path + "/cli/mf -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_mf.txt -dimensions 64 -sample_times 10 -negative_samples 5 -alpha 0.025 -reg 0.01 -threads 1"
    print(cmd)
    os.system(cmd)

    # bpr
    cmd = smore_path + "/cli/bpr -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_bpr.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # warp
    cmd = smore_path + "/cli/warp -train " + data_path + "/net.txt -save " + data_path + \
          "/rep_warp.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)

    # hoprec
    cmd = smore_path + "/cli/hoprec -train " + data_path + "/net.txt -save " + data_path + "/rep_hoprec.txt -field " \
         + data_path + "/field.txt -dimensions 64 -sample_times 10 -alpha 0.025 -threads 1"
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set, data_path):
    # nemf
    # load word vectors file
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_nemf.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "nemf.tsv")

    # nerank
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_nerank.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "nerank.tsv")

    # deepwalk
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_dw.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "deepwalk.tsv")

    # walklets
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_wl.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "walklets.tsv")

    # line order=1
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line1.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "line1.tsv")

    # line order=2
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line2.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "line2.tsv")

    # hpe
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hpe.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "hpe.tsv")

    # app
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_app.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "app.tsv")

    # mf
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_mf.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "mf.tsv")

    # bpr
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_bpr.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "bpr.tsv")

    # warp
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_warp.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "warp.tsv")

    # hoprec
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hoprec.txt"), binary=False)
    _recommend(user_set, cand_set, word_vectors, data_path, "hoprec.tsv")


def _recommend(user_set, cand_set, word_vectors, data_path, recommend_file):
    with open(os.path.join(data_path, recommend_file), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(str(user), str(item)) \
                    if str(user) in word_vectors and str(item) in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


def main():
    data_path = "../../data/douban/baseline"
    smore_path = "../../../smore"
    test_ratio = 0.25

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    user_set, cand_set = preprocess_net(data_path, test_ratio)
    run_model(smore_path, data_path)
    recommend(user_set, cand_set, data_path)


if __name__ == "__main__":
    main()
