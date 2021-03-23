import numpy as np
import tensorflow as tf
import argparse


class IGEModel:
    def __init__(self, num_users, num_items, n_sampled=100, embedding_dim=128, lr=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.n_sampled = n_sampled
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.softmax_w_u = tf.Variable(tf.truncated_normal((num_items, embedding_dim), stddev=0.1), name='softmax_w_u')
        self.softmax_w_v = tf.Variable(tf.truncated_normal((num_items, embedding_dim), stddev=0.1), name='softmax_w_v')
        self.softmax_b = tf.Variable(tf.zeros(num_items), name='softmax_b')
        self.inputs = self.input_init()
        self.embedding = self.embedding_init()
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--root_path", type=str, default='../../../data/amazon/')
    args = parser.parse_args()

    s_u = np.loadtxt(args.root_path + 's_u.csv', dtype=np.int32, delimiter='\t')


