import os
import pickle
from gensim.models import KeyedVectors
import numpy as np


def recommend(data_path, representation_size):
    user_set = pickle.load(open(os.path.join(data_path, 'user_set.pkl'), 'rb'))
    cand_set = pickle.load(open(os.path.join(data_path, 'cand_set.pkl'), 'rb'))
    user_items_train_map = pickle.load(open(os.path.join(data_path, 'user_items_train_map.pkl'), 'rb'))
    item_category_map = pickle.load(open(os.path.join(data_path, 'item_category_map.pkl'), 'rb'))

    # load word vectors file
    wv = KeyedVectors.load_word2vec_format(os.path.join(data_path, "wv.txt"), binary=False)
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "wordvecs.txt"), binary=False)
    doc_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "docvecs.txt"), binary=False)

    item_vectors_map = {}
    for item in word_vectors.vocab:
        item_vectors_map[item] = word_vectors[item] + (doc_vectors["*dt_" + item_category_map[int(item)]]
                                                       if int(item) in item_category_map else np.zeros(representation_size))

    with open(os.path.join(data_path, "deepwalk.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([wv.similarity(str(cand), str(item)) for item in user_items_train_map[user]]) \
                    if str(cand) in wv else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join([str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

    with open(os.path.join(data_path, "category2vec.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([item_vectors_map[str(cand)].dot(item_vectors_map[str(item)].T)
                             for item in user_items_train_map[user]]) if str(cand) in word_vectors else 0
                item_score_list.append((cand, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(str(user) + "\t")
            recommend.write(
                ",".join(
                    [str(item_score[0]) + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


def main():
    data_path = "../../data/douban"
    representation_size = 64

    recommend(data_path, representation_size)


if __name__ == "__main__":
    main()