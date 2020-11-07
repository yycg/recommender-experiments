from item_based_preprocess import preprocess_net
from build_graph import build_graph
from category_deepwalk import process
from recommend import recommend

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
test_ratio = 0.25


if __name__ == "__main__":
    preprocess_net(data_path, test_ratio)
    build_graph(data_path)
    process(data_path, input, category_input, wv_output, docvecs_output, wordvecs_output, undirected, number_walks,
            walk_length, representation_size, window_size, workers, seed)
    recommend(data_path, representation_size)
