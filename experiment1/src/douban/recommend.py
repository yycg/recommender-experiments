import os
import pickle
from gensim.models import KeyedVectors
from gensim import matutils
import numpy as np
from numpy import dot


def recommend(data_path, representation_size, deepwalk_recommend_list, category2vec_recommend_list):
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
        item_vectors_map[item] = word_vectors[item] + (doc_vectors["*dt_" + str(item_category_map[int(item)])]
           # if int(item) in item_category_map else np.zeros(representation_size))
           if int(item) in item_category_map and "*dt_" + str(item_category_map[int(item)]) in doc_vectors
           else np.zeros(representation_size))

    with open(os.path.join(data_path, deepwalk_recommend_list), "w") as recommend:
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

    with open(os.path.join(data_path, category2vec_recommend_list), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                score = sum([dot(matutils.unitvec(item_vectors_map[str(cand)]),
                                 matutils.unitvec(item_vectors_map[str(item)]))
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
    deepwalk_recommend_list = "deepwalk.tsv"
    category2vec_recommend_list = "category2vec.tsv"

    recommend(data_path, representation_size, deepwalk_recommend_list, category2vec_recommend_list)


if __name__ == "__main__":
    main()
