import os
import pandas as pd
from sqlalchemy import create_engine
from gensim.models import KeyedVectors
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def preprocess_net(data_path, test_ratio):
    print("Fetch data")
    engine = create_engine(
        "mysql+pymysql://douban_readonly:douban_readonly@10.105.240.25:3306/douban_beijing_2018?charset=utf8")
    sql = "select * from eventuser"
    eventuser = pd.read_sql_query(sql, engine)
    eventuser = eventuser.loc[eventuser["user_type"] == "participant"]

    sql = "select * from event"
    event = pd.read_sql_query(sql, engine)

    print("Preprocess data")
    event_set = set()
    for index, row in event.iterrows():
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

    category_items_map = {}
    for index, row in event.iterrows():
        item = row["id"]
        category = row["owner_id"]
        if item in item_set:
            category_items_map.setdefault(category, [])
            category_items_map[category].append(item)

    print("Write data to files")
    with open(os.path.join(data_path, "net.txt"), "w") as net:
        # user-item pair
        for index, row in eventuser_train.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            net.write(str(user) + " " + str(item) + " 1\n")
        # property-item pair
        for category, items in category_items_map.items():
            for item in items:
                net.write(str(category) + " " + str(item) + " 1\n")

    with open(os.path.join(data_path, "field.txt"), "w") as field:
        for user in user_set:
            field.write(str(user) + " u\n")
        for item in item_set:
            field.write(str(item) + " i\n")
        for category in category_items_map:
            field.write(str(category) + " p\n")

    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "w") as test:
        for index, row in eventuser_test.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            test.write(str(user) + "\t" + str(item) + "\n")

    return test_user_set, test_cand_set


def run_model(smore_path, data_path, sample_times, walk_steps, alpha, dimensions):
    # ccse
    cmd = smore_path + "/cli/ccse -train " + data_path + "/net.txt -save " + data_path + "/rep_ccse.txt -field " \
          + data_path + "/field.txt -dimensions {3} -sample_times {0} -walk_steps {1} -alpha {2} -threads 1" \
              .format(sample_times, walk_steps, alpha, dimensions)
    print(cmd)
    os.system(cmd)


def recommend(user_set, cand_set, data_path):
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

def main(args):
    test_ratio = 0.25
    sample_times = args.sample_times
    walk_steps = args.walk_steps
    alpha = args.alpha
    dimensions = args.dimensions
    data_path = os.path.join("../../data/douban/ccse", "sample_times{}".format(sample_times),
                             "walk_steps{}".format(walk_steps), "alpha{}".format(alpha),
                             "dimensions{}".format(dimensions)) if args.data_path is None else args.data_path
    smore_path = args.smore_path

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    user_set, cand_set = preprocess_net(data_path, test_ratio)
    run_model(smore_path, data_path, sample_times, walk_steps, alpha, dimensions)
    recommend(user_set, cand_set, data_path)


if __name__ == "__main__":
    parser = ArgumentParser("ccse",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--data_path")
    parser.add_argument("--smore_path", default="../../../smore")
    parser.add_argument("--sample_times", default=40, type=int)
    parser.add_argument("--walk_steps", default=5, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--dimensions", default=128, type=int)
    args = parser.parse_args()

    main(args)
