import os
import itertools

data_path = "../../data/douban"

if __name__ == "__main__":
    with open(os.path.join(data_path, "user_item_list.txt"), "r") as reader, \
            open(os.path.join(data_path, "user_item_edge.txt"), "w") as writer:
        for line in reader:
            items = line.split(" ")
            edges = itertools.permutations(items, 2)
            for edge in edges:
                writer.write(" ".join(edge) + "\n")

    with open(os.path.join(data_path, "category_item_list.txt"), "r") as reader, \
            open(os.path.join(data_path, "category_item_edge.txt"), "w") as writer:
        for line in reader:
            items = line.split(" ")
            category = items[0]
            items = items[1:]

