import os
import pandas as pd
from sqlalchemy import create_engine

data_path = "../../data/douban"
test_ratio = 0.25

def preprocess_net():
    print("Fetch data")
    engine = create_engine(
        "mysql+pymysql://douban_readonly:douban_readonly@10.112.207.78:3306/douban_beijing_2018?charset=utf8")
    sql = "select * from eventuser"
    eventuser = pd.read_sql_query(sql, engine)
    eventuser = eventuser.loc[eventuser["user_type"] == "participant"]

    sql = "select id, category from event"
    event = pd.read_sql_query(sql, engine)
    event = event.loc[:, ["id", "category"]]

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
        if 5 <= user_occurrence_count_map[user] < 20 and 5 <= item_occurrence_count_map[item] < 100:
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
        category = row["category"]
        category_items_map.setdefault(category, [])
        category_items_map[category].append(item)

    print("Write data to files")
    with open(os.path.join(data_path, "user_item_list.txt"), "w") as net:
        for user, maps in user_items_map.items():
            for i in range(len(maps)):
                if i < (1 - test_ratio) * len(maps):
                    net.write(str(maps[i]["item"]) + " ")
            net.write("\n")

    with open(os.path.join(data_path, "category_item_list.txt"), "w") as file:
        for category, items in category_items_map.items():
            file.write(category + " " + " ".join([str(item) for item in items]) + "\n")

    with open(os.path.join(data_path, "user-event-rsvp_test.tsv"), "w") as test:
        for index, row in eventuser_test.iterrows():
            user = row["user_id"]
            item = row["event_id"]
            test.write(str(user) + "\t" + str(item) + "\n")


if __name__ == "__main__":
    preprocess_net()
