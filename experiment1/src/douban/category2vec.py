from item_based_preprocess import preprocess_net
from build_graph import build_graph
from category_deepwalk import process
from recommend import recommend
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


if __name__ == "__main__":
    parser = ArgumentParser("category2vec",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--data_path")
    parser.add_argument("--undirected", default=True, type=bool)
    parser.add_argument("--number_walks", default=10, type=int)
    parser.add_argument("--walk_length", default=40, type=int)
    parser.add_argument("--representation_size", default=64, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--workers", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    undirected = args.undirected
    number_walks = args.number_walks
    walk_length = args.walk_length
    representation_size = args.representation_size
    window_size = args.window_size
    workers = args.workers
    seed = args.seed
    test_ratio = 0.25

    data_path = os.path.join("../../data/douban/category2vec", "number_walks{}".format(number_walks),
                             "walk_length{}".format(walk_length), "representation_size{}".format(representation_size),
                             "window_size{}".format(window_size)) if args.data_path is None else args.data_path
    input = "user_item_edge.txt"
    category_input = "category_item_edge.txt"
    wv_output = "wv.txt"
    docvecs_output = "docvecs.txt"
    wordvecs_output = "wordvecs.txt"
    deepwalk_recommend_list = "deepwalk.tsv"
    category2vec_recommend_list = "category2vec.tsv"

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    preprocess_net(data_path, test_ratio)
    build_graph(data_path)
    process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed)
    recommend(data_path, representation_size, deepwalk_recommend_list, category2vec_recommend_list)
