from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def run_ccse(sample_times, walk_steps, alpha, dimensions, lambda1, lambda2, lambda_factor):
    cmd = "python ccse.py --sample_times {0} --walk_steps {1} --alpha {2} --dimensions {3} --lambda1 {4} " \
          "--lambda2 {5} --lambda_factor {6}"\
        .format(sample_times, walk_steps, alpha, dimensions, lambda1, lambda2, lambda_factor)
    print(cmd)
    os.system(cmd)
    return cmd


def main():
    # https://www.jianshu.com/p/b9b3d66aa0be
    all_task = []
    executor = ThreadPoolExecutor(max_workers=20)
    for sample_times in [20, 40, 60, 80, 100]:
        for walk_steps in [40, 60, 80]:
            for alpha in [0.0025, 0.01, 0.025, 0.1]:
                for dimensions in [64, 128, 256, 512]:
    # for sample_times in [40]:
    #     for walk_steps in [5, 40]:
    #         for alpha in [0.01]:
    #             for dimensions in [128]:
    #                 for lambda1 in [0.0025, 0.01, 0.025, 0.1, 0.25]:
    #                     for lambda2 in [0.0025, 0.01, 0.025, 0.1, 0.25]:
    #                         for lambda_factor in [0.0025, 0.01, 0.025, 0.1, 0.25, 1, 2.5, 10, 25]:
                    for lambda1 in [0.25]:
                        for lambda2 in [0.0025]:
                            for lambda_factor in [0.25]:
                                all_task.append(executor.submit(run_ccse, sample_times, walk_steps, alpha, dimensions,
                                                                lambda1, lambda2, lambda_factor))
    for future in as_completed(all_task):
        data = future.result()
        print("{} succeed".format(data))


if __name__ == '__main__':
    main()
