import json

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import argparse
import networkx as nx
import os

from walker import RandomWalker


class HIGE:
    """1. node2vec hierarchy
       2. node2vec attributes high-order
       3. attention, cold start
    """
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0,
                 use_random_leap=True, item_attr_map=None, attr_items_map=None, r=0.05,
                 use_hierarchical_structure=False, item_categories_map=None,
                 category_category_children_map=None, category_item_children_map=None, h=0.5, h2=0.5):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, item_attr_map=item_attr_map, attr_items_map=attr_items_map, p=p, q=q,
            use_rejection_sampling=use_rejection_sampling, use_random_leap=use_random_leap, r=r, h=h, h2=h2,
            item_categories_map=item_categories_map, category_category_children_map=category_category_children_map,
            category_item_children_map=category_item_children_map, use_hierarchical_structure=use_hierarchical_structure)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


def output_embeddings(embeddings):
    with open("../../../data/amazon/embedding/node2vec.embed", "w") as file:
        num_nodes = len(embeddings)
        for node, emb in embeddings.items():
            embedding_size = len(emb)
            break

        file.write(str(num_nodes) + " " + str(embedding_size) + "\n")
        for node, emb in embeddings.items():
            file.write(str(node) + " " + " ".join([str(e) for e in emb]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='../../../data/amazon/')
    args = parser.parse_args()

    # https://blog.csdn.net/gulaixiangjuejue/article/details/101380572
    # 加create_using=nx.DiGraph()是有向图，不加是无向图
    G = nx.read_edgelist(os.path.join(args.data_path, 'net.txt'),
                         nodetype=None, data=[('weight', int)])

    sku_side_info = np.loadtxt(args.data_path + 'sku_side_info.csv', dtype=np.int32, delimiter='\t')
    item_attr_map = {}
    attr_items_map = {}
    for i in range(len(sku_side_info)):
        item = sku_side_info[i][0]
        side_info = sku_side_info[i][1]
        item_attr_map[str(item)] = str(side_info)
        attr_items_map.setdefault(str(side_info), [])
        attr_items_map[str(side_info)].append(str(item))

    sku_category = pd.read_csv(args.data_path + 'sku_category.csv', sep="\t", header=None, names=["item", "categories"])
    category_category_children = pd.read_csv(args.data_path + 'category_category_children.csv', sep="\t", header=None,
                                             names=["category", "category_children"])
    category_item_children = pd.read_csv(args.data_path + 'category_item_children.csv', sep="\t", header=None,
                                         names=["category", "item_children"])
    item_categories_map = {}
    for index, row in sku_category.iterrows():
        item = str(row['item'])
        categories = [str(s) for s in json.loads(row['categories'])]
        item_categories_map[item] = categories
    category_category_children_map = {}
    for index, row in category_category_children.iterrows():
        category = str(row['category'])
        category_children = row['category_children'].split(',')
        category_category_children_map[category] = category_children
    category_item_children_map = {}
    for index, row in category_item_children.iterrows():
        category = str(row['category'])
        item_children = row['item_children'].split(',')
        category_item_children_map[category] = item_children

    model = HIGE(G, item_attr_map=item_attr_map, attr_items_map=attr_items_map, item_categories_map=item_categories_map,
                 category_category_children_map=category_category_children_map,
                 category_item_children_map=category_item_children_map, walk_length=10, num_walks=80,
                 p=0.25, q=4, workers=1, use_rejection_sampling=0, use_random_leap=True, r=0.05, h=0.05, h2=0.05,
                 use_hierarchical_structure=True)
    model.train(window_size=5, iter=3)
    embeddings=model.get_embeddings()
    output_embeddings(embeddings)
