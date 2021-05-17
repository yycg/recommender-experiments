from gensim.models import Word2Vec
import numpy as np
import argparse
import networkx as nx
import os

from walker import RandomWalker


class HIGE:
    """1. node2vec hierarchy
       2. node2vec attributes high-order
       3. cold start
    """
    def __init__(self, graph, item_side_info_map, walk_length, num_walks, p=1.0, q=1.0, workers=1,
                 use_rejection_sampling=0, use_random_leap=True, r=0.05):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, item_side_info_map=item_side_info_map, p=p, q=q, use_rejection_sampling=use_rejection_sampling,
            use_random_leap=use_random_leap, r=r)

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
    item_side_info_map = {}
    for i in range(len(sku_side_info)):
        item = sku_side_info[i][0]
        side_info = sku_side_info[i][1]
        item_side_info_map[str(item)] = str(side_info)
    model = HIGE(G, item_side_info_map=item_side_info_map, walk_length=10, num_walks=80,
                 p=0.25, q=4, workers=1, use_rejection_sampling=0, use_random_leap=True, r=0.5)
    model.train(window_size=5, iter=3)
    embeddings=model.get_embeddings()
    output_embeddings(embeddings)