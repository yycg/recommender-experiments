import os
import pickle
from gensim.models import KeyedVectors

data_path = "../../data/douban"


def recommend():
    user_set = pickle.load(open(os.path.join(data_path, 'user_set.pkl'), 'rb'))
    cand_set = pickle.load(open(os.path.join(data_path, 'cand_set.pkl'), 'rb'))
    user_items_train_map = pickle.load(open(os.path.join(data_path, 'user_items_train_map.pkl'), 'rb'))
    item_category_map = pickle.load(open(os.path.join(data_path, 'item_category_map.pkl'), 'rb'))

    # load word vectors file
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "wordvecs.txt"), binary=False)
    doc_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "docvecs.txt"), binary=False)

    item_vectors_map = {}
    for item in word_vectors:
        item_vectors_map[item] = word_vectors[item] + doc_vectors[item_category_map[item]]

    with open(os.path.join(data_path, "category2vec.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for cand in cand_set:
                # score = word_vectors.similarity(user, cand) \
                #     if user in word_vectors and cand in word_vectors else 0
                score = sum([item_vectors_map[cand].dot(item_vectors_map[item].T) for item in user_items_train_map[user]])
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(
                ",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")


if __name__ == "__main__":
    recommend()
