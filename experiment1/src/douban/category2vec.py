from item_based_preprocess import preprocess_net
from build_graph import build_graph
from category_deepwalk import process
from recommend import recommend
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


if __name__ == "__main__":
    parser = ArgumentParser("category2vec",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--data_path", default="../../data/douban")
    parser.add_argument("--input", default="user_item_edge.txt")
    parser.add_argument("--category_input", default="category_item_edge.txt")
    parser.add_argument("--wv_output", default="wv.txt")
    parser.add_argument("--docvecs_output", default="docvecs.txt")
    parser.add_argument("--wordvecs_output", default="wordvecs.txt")
    parser.add_argument("--undirected", default=True, type=bool)
    parser.add_argument("--number_walks", default=10, type=int)
    parser.add_argument("--walk_length", default=40, type=int)
    parser.add_argument("--representation_size", default=64, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--workers", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test_ratio", default=0.25, type=float)
    parser.add_argument("--deepwalk_recommend_list", default="deepwalk.tsv")
    parser.add_argument("--category2vec_recommend_list", default="category2vec.tsv")
    args = parser.parse_args()

    data_path = args.data_path
    input = args.input
    category_input = args.category_input
    wv_output = args.wv_output
    docvecs_output = args.docvecs_output
    wordvecs_output = args.wordvecs_output
    undirected = args.undirected
    number_walks = args.number_walks
    walk_length = args.walk_length
    representation_size = args.representation_size
    window_size = args.window_size
    workers = args.workers
    seed = args.seed
    test_ratio = args.test_ratio
    deepwalk_recommend_list = args.deepwalk_recommend_list
    category2vec_recommend_list = args.category2vec_recommend_list

    preprocess_net(data_path, test_ratio)
    build_graph(data_path)
    process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed)
    recommend(data_path, representation_size, deepwalk_recommend_list, category2vec_recommend_list)
