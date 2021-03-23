import os
import pickle
from gensim.models import KeyedVectors
from gensim import matutils
import numpy as np
from numpy import dot


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


def recommend(data_path, user_set, cand_set):
    # load word vectors file
    wv_user = KeyedVectors.load_word2vec_format(os.path.join(data_path, "embedding", "IGEUser.embed"), binary=False)
    wv_item = KeyedVectors.load_word2vec_format(os.path.join(data_path, "embedding", "IGEItem.embed"), binary=False)

    with open(os.path.join(data_path, "IGE.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = dot(matutils.unitvec(wv_user[str(user)]), matutils.unitvec(wv_item[str(item)])) \
                    if str(user) in wv_user and str(item) in wv_item else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join(
                    [str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


if __name__ == "__main__":
    data_path = "../../../data/amazon"
    user_set, cand_set = get_user_cand_set(data_path)
    recommend(data_path, user_set, cand_set)
