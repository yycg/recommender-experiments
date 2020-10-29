#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

# https://github.com/phanein/deepwalk/issues/29
from deepwalk import graph
from gensim.models import Word2Vec


data_path = "../../data/douban"
input = os.path.join(data_path, "user_item_edge.txt")
category_input = os.path.join(data_path, "category_item_edge.txt")
output = os.path.join(data_path, "item_embedding.txt")
undirected = True
number_walks = 10
walk_length = 40
representation_size = 64
window_size = 5
workers = 1
seed = 0


def process():
    G = graph.load_edgelist(input, undirected=undirected)

    category_graph_map = load_category_edgelist(category_input, undirected=undirected)

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    category_walks_map = build_catagory_deepwalk_corpus(category_graph_map, number_walks=number_walks,
                                                    path_length=walk_length, alpha=0, rand=random.Random(seed))

    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                     workers=workers)

    model.wv.save_word2vec_format(output)


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


def build_catagory_deepwalk_corpus(category_graph_map, num_paths, path_length, alpha=0, rand=random.Random(0)):
    category_walks_map = {}

    for category, G in category_graph_map.items():
        walks = []

        nodes = list(G.nodes())

        for cnt in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

        category_graph_map[category] = walks

    return category_walks_map


if __name__ == "__main__":
    process()
