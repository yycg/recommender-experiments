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

    sql = "select id, category from event"
    event = pd.read_sql_query(sql, engine)
    event = event.loc[:, ["id", "category"]]

    print("Preprocess data")
    user_set = set()
    item_set = set()
    user_items_map = {}
    for index, row in eventuser.iterrows():
        user = row["user_id"]
        item = row["event_id"]
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


def run_model():
    # ccse
    cmd = CSE_path + "/cli/ccse -train " + data_path + "/net.txt -save " + data_path + "/rep_ccse.txt -field " \
          + data_path + "/field.txt -dimensions 128 -sample_times {0} -walk_steps {1} -alpha {2} -threads 1" \
              .format(sample_times, walk_steps, alpha)
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set):
    # ccse
    # load word vectors file
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_ccse.txt"), binary=False)

    with open(os.path.join(data_path, "ccse.tsv"), "w") as recommend:
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
    data_path = "../../data/douban/baseline"
    test_ratio = 0.25
    CSE_path = "../../../smore"
    sample_times = 40
    walk_steps = 5
    alpha = 0.01

    user_set, cand_set = preprocess_net(data_path, test_ratio)
    run_model()
    recommend(user_set, cand_set)
