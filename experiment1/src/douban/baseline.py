import os
import pandas as pd
from sqlalchemy import create_engine
from gensim.models import KeyedVectors

from build_graph import build_graph


def preprocess_net(data_path, test_ratio):
    print("Fetch data")
    engine = create_engine(
        "mysql+pymysql://douban_readonly:douban_readonly@10.112.207.78:3306/douban_beijing_2018?charset=utf8")
    sql = "select * from eventuser"
    eventuser = pd.read_sql_query(sql, engine)
    eventuser = eventuser.loc[eventuser["user_type"] == "participant"]

    print("Preprocess data")
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
        if 5 <= user_occurrence_count_map[user] < 20 and 5 <= item_occurrence_count_map[item] < 20:
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
    with open(os.path.join(data_path, "user_item_list.txt"), "w") as net:
        for user, maps in user_items_map.items():
            for i in range(len(maps)):
                if i < (1 - test_ratio) * len(maps):
                    net.write(str(maps[i]["item"]) + " ")
            net.write("\n")

    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "w") as test:
        for index, row in eventuser_test.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            test.write(str(user) + "\t" + str(item) + "\n")

    user_items_train_map = {}
    for index, row in eventuser_train.iterrows():
        user = row["user_id"]
        item = row["event_id"]
        user_items_train_map.setdefault(user, [])
        user_items_train_map[user].append(item)

    return test_user_set, test_cand_set, user_items_train_map


def run_model(data_path, CSE_path, sample_times, walk_steps, alpha, item2vec_path):
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

    # item2vec
    cmd = item2vec_path + "/fastText-0.9.1/fasttext skipgram -input " + data_path + "/net.txt -output " + data_path + \
          "item2vec.txt -minCount 5 -epoch 50 -neg 100"
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set, data_path, user_items_train_map):
    # deepwalk
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_dw.txt"), binary=False)

    with open(os.path.join(data_path, "deepwalk.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


    # walklets
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_wl.txt"), binary=False)

    with open(os.path.join(data_path, "walklets.tsv"), "w") as recommend:
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

    # line order=1
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line1.txt"), binary=False)

    with open(os.path.join(data_path, "line1.tsv"), "w") as recommend:
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

    # line order=2
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_line2.txt"), binary=False)

    with open(os.path.join(data_path, "line2.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # hpe
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hpe.txt"), binary=False)

    with open(os.path.join(data_path, "hpe.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # app
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_app.txt"), binary=False)

    with open(os.path.join(data_path, "app.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # mf
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_mf.txt"), binary=False)

    with open(os.path.join(data_path, "mf.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # bpr
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_bpr.txt"), binary=False)

    with open(os.path.join(data_path, "bpr.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # warp
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_warp.txt"), binary=False)

    with open(os.path.join(data_path, "warp.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # hoprec
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_hoprec.txt"), binary=False)

    with open(os.path.join(data_path, "hoprec.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([word_vectors.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    # item2vec
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "item2vec.txt"), binary=False)

    with open(os.path.join(data_path, "item2vec.tsv"), "w") as recommend:
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


def main():
    data_path = "../../data/douban/baseline"
    test_ratio = 0.25
    CSE_path = "../../smore"
    sample_times = 40
    walk_steps = 5
    alpha = 0.01
    item2vec_path = "../../item2vec"

    user_set, cand_set, user_items_train_map = preprocess_net(data_path, test_ratio)
    build_graph(data_path)
    run_model(data_path, CSE_path, sample_times, walk_steps, alpha, item2vec_path)
    recommend(user_set, cand_set, data_path, user_items_train_map)


if __name__ == "__main__":
    main()
