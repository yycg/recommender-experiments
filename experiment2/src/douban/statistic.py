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

    num_user = len(user_set)
    num_item = len(item_set)
    num_rsvp = sum(len(items) for user, items in user_items_map.items())
    print("# of users", num_user)
    print("# of items", num_item)
    print("# of rsvps", num_rsvp)


def main():
    test_ratio = 0.25
    data_path = os.path.join("../../data/douban")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    preprocess_net(data_path, test_ratio)


if __name__ == "__main__":
    main()
