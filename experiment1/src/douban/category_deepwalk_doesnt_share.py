#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

# https://github.com/phanein/deepwalk/issues/29
from deepwalk import graph
from gensim.models import Word2Vec, Category2Vec
from gensim.models.category2vec import TaggedLineDocument


def process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed):
    G = graph.load_edgelist(os.path.join(data_path, input), undirected=undirected)
    category_graph_map = load_category_edgelist(os.path.join(data_path, category_input), undirected=undirected)

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    category_walks = build_category_deepwalk_corpus(category_graph_map, num_paths=number_walks,
                                                    path_length=walk_length, alpha=0, rand=random.Random(seed))

    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                     workers=workers)

    model.wv.save_word2vec_format(os.path.join(data_path, wv_output))

    model = Category2Vec(walks, size=representation_size, window=window_size, min_count=0, dm=0, hs=0,
                         workers=workers, category_documents=category_walks)
    # model.wv.save_word2vec_format(os.path.join(data_path, wv_output))
    model.docvecs.save_word2vec_format(os.path.join(data_path, docvecs_output))
    model.wordvecs.save_word2vec_format(os.path.join(data_path, wordvecs_output))


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
    return TaggedLineDocument(category_graph_map, num_paths, path_length, alpha, rand)


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
