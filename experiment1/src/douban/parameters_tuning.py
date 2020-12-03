from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def run_category2vec(number_walks, walk_length, representation_size, window_size, lambda_factor):
    cmd = "python category2vec.py --number_walks {0} --walk_length {1} --representation_size {2} --window_size {3} " \
          "--lambda_factor {4} " \
        .format(number_walks, walk_length, representation_size, window_size, lambda_factor)
    print(cmd)
    os.system(cmd)
    return cmd


def main():
    # https://www.jianshu.com/p/b9b3d66aa0be
    all_task = []
    executor = ThreadPoolExecutor(max_workers=20)
    # for number_walks in [10, 40, 80]:  # walk_times
    #     for walk_length in [40, 60, 80]:  # walk_steps
    #         for representation_size in [64, 128, 256, 512]:  # dimensions
    #             for window_size in [2, 3, 4, 5, 6, 8, 10]:  # window_size
    #                 for lambda_factor in [0.0025, 0.01, 0.025, 0.1, 0.25, 1, 2.5, 10, 25]:
    # for number_walks in [20, 60, 100]:  # walk_times
    for number_walks in [2, 4, 6, 8]:  # walk_times
        for walk_length in [40]:  # walk_steps
            for representation_size in [256]:  # dimensions
                for window_size in [3]:  # window_size
                    for lambda_factor in [10]:
                        all_task.append(executor.submit(run_category2vec, number_walks, walk_length,
                                                        representation_size, window_size, lambda_factor))
    for future in as_completed(all_task):
        data = future.result()
        print("{} succeed".format(data))


if __name__ == '__main__':
    main()
