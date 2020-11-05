

def recommend():
    # load word vectors file
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(data_path, "rep_nemf.txt"), binary=False)

    with open(os.path.join(data_path, "nemf.tsv"), "w") as recommend:
        for user in user_set:
            item_score_list = []
            for item in cand_set:
                score = word_vectors.similarity(user, item) \
                    if user in word_vectors and item in word_vectors else 0
                item_score_list.append((item, score))
            item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

            recommend.write(user + "\t")
            recommend.write(
                ",".join([item_score[0] + ":" + str(item_score[1]) for item_score in item_score_list[:100]]) + "\n")

if __name__ == "__main__":
    recommend(user_set, cand_set)
