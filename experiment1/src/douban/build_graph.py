import os
import itertools


def build_graph(data_path):
    with open(os.path.join(data_path, "user_item_list.txt"), "r") as reader, \
            open(os.path.join(data_path, "user_item_edge.txt"), "w") as writer:
        for line in reader:
            items = line.strip().split(" ")
            edges = itertools.permutations(items, 2)
            for edge in edges:
                writer.write(" ".join(edge) + "\n")

    with open(os.path.join(data_path, "category_item_list.txt"), "r") as reader, \
            open(os.path.join(data_path, "category_item_edge.txt"), "w") as writer:
        for line in reader:
            items = line.strip().split(" ")
            category = items[0]
            items = items[1:]
            edges = itertools.permutations(items, 2)
            for edge in edges:
                writer.write(category + " " + " ".join(edge) + "\n")


def main():
    data_path = "../../data/douban"

    build_graph(data_path)


if __name__ == "__main__":
    main()
