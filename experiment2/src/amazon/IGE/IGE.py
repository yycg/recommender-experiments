import numpy as np
import tensorflow as tf
import argparse
import time
from utils import *

# TODO: 两个softmax拼接
class IGEModel:
    def __init__(self, num_users, num_items, n_sampled=100, embedding_dim=128, lr=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.n_sampled = n_sampled
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.softmax_w_u = tf.Variable(tf.truncated_normal((num_items, embedding_dim), stddev=0.1), name='softmax_w_u')
        # self.softmax_w_v = tf.Variable(tf.truncated_normal((num_items, embedding_dim), stddev=0.1), name='softmax_w_v')
        self.softmax_b = tf.Variable(tf.zeros(num_items), name='softmax_b')
        self.inputs = self.input_init()
        self.embedding = self.embedding_init()
        self.alpha_embedding = tf.Variable(tf.random_uniform((num_items, 2), -1, 1))
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

    def input_init(self):
        input_list = []
        # user
        input_col = tf.placeholder(tf.int32, [None], name='inputs_' + str(1))
        input_list.append(input_col)
        # item
        input_col = tf.placeholder(tf.int32, [None], name='inputs_' + str(2))
        input_list.append(input_col)
        # label
        input_list.append(tf.placeholder(tf.int32, shape=[None, 1], name='label'))
        return input_list

    def embedding_init(self):
        cat_embedding_vars = []
        # user
        embedding_var = tf.Variable(tf.random_uniform((self.num_users, self.embedding_dim), -1, 1), name='embedding'+str(1),
                                    trainable=True)
        cat_embedding_vars.append(embedding_var)
        # item
        embedding_var = tf.Variable(tf.random_uniform((self.num_items, self.embedding_dim), -1, 1),
                                    name='embedding' + str(2),
                                    trainable=True)
        cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        embed_list = []
        num_embed_list = []
        # user
        cat_embed = tf.nn.embedding_lookup(self.embedding[0], self.inputs[0])
        embed_list.append(cat_embed)
        # item
        cat_embed = tf.nn.embedding_lookup(self.embedding[1], self.inputs[1])
        embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def make_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w_u,
            biases=self.softmax_b,
            labels=self.inputs[-1],
            inputs=self.merge_emb,
            num_sampled=self.n_sampled,
            num_classes=self.num_items,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.inputs[-1], tf.int64),
                num_true=1,
                num_sampled=self.n_sampled,
                unique=True,
                range_max=self.num_items
            )
        ))
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    # 采样次数
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='../../../data/amazon/')
    # sku_side_info.csv的列数，第一列是物品id，后面列是特征id
    # parser.add_argument("--num_feat", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputUserEmbedFile", type=str, default='../../../data/amazon/embedding/IGEUser.embed')
    parser.add_argument("--outputItemEmbedFile", type=str, default='../../../data/amazon/embedding/IGEItem.embed')
    args = parser.parse_args()

    train = np.loadtxt(args.root_path + 'train.tsv', dtype=np.int32, delimiter='\t')
    test = np.loadtxt(args.root_path + 'user-event-rsvp_test.tsv', dtype=np.int32, delimiter='\t')
    num_users = 0
    num_items = 0
    for i in range(len(train)):
        num_users = max(num_users, train[i][0])
        num_items = max(num_items, train[i][1])
    for i in range(len(test)):
        num_users = max(num_users, test[i][0])
        num_items = max(num_items, test[i][1])
    num_users += 1
    num_items += 1

    s_u = np.loadtxt(args.root_path + 's_u.csv', dtype=np.int32, delimiter=' ')
    IGE = IGEModel(num_users, num_items, n_sampled=args.n_sampled, embedding_dim=args.embedding_dim, lr=args.lr)

    # init model
    print('init...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()

    max_iter = len(s_u) // args.batch_size * args.epochs
    for iter in range(max_iter):
        iteration += 1
        batch_features, batch_labels = next(graph_context_batch_iter(s_u, args.batch_size))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(IGE.inputs[:-1])}
        feed_dict[IGE.inputs[-1]] = batch_labels
        _, train_loss = sess.run([IGE.train_op, IGE.cost], feed_dict=feed_dict)

        loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration * args.batch_size // len(s_u)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            loss = 0
            start = time.time()

    print('optimization finished...')
    saver = tf.train.Saver()
    saver.save(sess, "checkpoints/IGE")

    # user embedding
    test_s_u = np.zeros((num_users, 2))
    test_s_u[:, 0] = range(num_users)
    feed_dict_test = {input_col: list(test_s_u[:, i]) for i, input_col in enumerate(IGE.inputs[:-1])}
    feed_dict_test[IGE.inputs[-1]] = np.zeros((num_users, 1), dtype=np.int32)
    embedding_result = sess.run(IGE.embedding, feed_dict=feed_dict_test)
    print('saving embedding result...')
    write_embedding(embedding_result, args.outputUserEmbedFile, args.embedding_dim)

    # item embedding
    test_s_u = np.zeros((num_items, 2))
    test_s_u[:, 0] = range(num_items)
    feed_dict_test = {input_col: list(test_s_u[:, i]) for i, input_col in enumerate(IGE.inputs[:-1])}
    feed_dict_test[IGE.inputs[-1]] = np.zeros((num_items, 1), dtype=np.int32)
    embedding_result = sess.run(IGE.embedding, feed_dict=feed_dict_test)
    print('saving embedding result...')
    write_embedding(embedding_result, args.outputItemEmbedFile, args.embedding_dim)
