from .item_based_preprocess import preprocess_net
from .build_graph import build_graph
from .category_deepwalk import process
from .recommend import recommend

if __name__ == "__main__":
    preprocess_net()
    build_graph()
    process()
    recommend()
