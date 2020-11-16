#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

# https://github.com/phanein/deepwalk/issues/29
from deepwalk import graph
from gensim.models import Word2Vec, Category2Vec, Doc2Vec
from gensim.models.category2vec import CategoryTaggedLineDocument, TaggedLineDocument


def process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed):
    G = graph.load_edgelist(os.path.join(data_path, input), undirected=undirected)
    category_graph_map = load_category_edgelist(os.path.join(data_path, category_input), undirected=undirected)

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    user_walks = build_deepwalk_corpus(G, num_paths=number_walks,
                                             path_length=walk_length, alpha=0, rand=random.Random(seed))
    category_walks = build_category_deepwalk_corpus(category_graph_map, num_paths=number_walks,
                                                    path_length=walk_length, alpha=0, rand=random.Random(seed))

    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=0,
                     workers=workers)

    model.wv.save_word2vec_format(os.path.join(data_path, wv_output))

    # model = Category2Vec(walks, size=representation_size, window=window_size, min_count=0, dm=0, hs=0,
    #                      workers=workers, category_documents=category_walks)
    # # model.wv.save_word2vec_format(os.path.join(data_path, wv_output))
    # model.docvecs.save_word2vec_format(os.path.join(data_path, docvecs_output))
    # model.wordvecs.save_word2vec_format(os.path.join(data_path, wordvecs_output))

    model = Doc2Vec(user_walks, size=representation_size, window_size=window_size, min_count=0, workers=workers,
                    hs=0, dm=0)
    model.wv.save_word2vec_format(os.path.join(data_path, wordvecs_output))
    start_alpha = 0.01
    infer_epoch = 1000
    with open(os.path.join(data_path, docvecs_output), "w") as output:
        # output.write(" ".join([str(x) for x in model.infer_vector_mod(category_walks, alpha=start_alpha, steps=infer_epoch)]) + "\n")
        category_doctag_map = model.infer_vector_mod(category_walks, alpha=start_alpha, steps=infer_epoch)
        output.write(str(len(category_doctag_map.keys())) + " " + str(representation_size) + "\n")
        for category, doctag in category_doctag_map.items():
            output.write(str(category) + " ")
            output.write(" ".join(str(x) for x in doctag["doctag_vectors"][0]) + "\n")


def load_category_edgelist(file_, undirected=True):
    category_graph_map = {}
    with open(file_) as f:
        for l in f:
            c, x, y = l.strip().split()[:3]
            x = int(x)
            y = int(y)
            category_graph_map.setdefault(c, graph.Graph())
            G = category_graph_map[c]
            G[x].append(y)
            if undirected:
                G[y].append(x)

    for category, G in category_graph_map.items():
        G.make_consistent()

    return category_graph_map


def build_category_deepwalk_corpus(category_graph_map, num_paths, path_length, alpha=0, rand=random.Random(0)):
    return CategoryTaggedLineDocument(category_graph_map, num_paths, path_length, alpha, rand)


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    return TaggedLineDocument(G, num_paths, path_length, alpha, rand)

    # walks = []
    #
    # nodes = list(G.nodes())
    #
    # for cnt in range(num_paths):
    #     rand.shuffle(nodes)
    #     for node in nodes:
    #         walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    #
    # return walks


def main():
    data_path = "../../data/douban"
    input = "user_item_edge.txt"
    category_input = "category_item_edge.txt"
    wv_output = "wv.txt"
    docvecs_output = "docvecs.txt"
    wordvecs_output = "wordvecs.txt"
    undirected = True
    number_walks = 10
    walk_length = 40
    representation_size = 64
    window_size = 5
    workers = 50
    seed = 0

    process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed)


if __name__ == "__main__":
    main()
