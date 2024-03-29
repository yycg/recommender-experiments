import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Input, Flatten, Concatenate, Dense, Lambda, Softmax
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.python.keras import initializers
import argparse
import random
import os
import math
import contextlib
import time

from alias import create_alias_table, alias_sample


class StubLogger(object):
    def __getattr__(self, name):
        return self.log_print

    def log_print(self, msg, *args):
        print(msg % args)


LOGGER = StubLogger()
LOGGER.info("Hello %s!", "world")


@contextlib.contextmanager
def elapsed_timer(message):
    start_time = time.time()
    yield
    LOGGER.info(message.format(time.time() - start_time))


def create_model(num_users, num_items, embedding_size):
    """Reference: Tensorflow Word2Vec tutorial
    https://www.tensorflow.org/tutorials/text/word2vec
    两种输入同时训练
    https://blog.csdn.net/xiaoxiao133/article/details/79653954
    keras share weights between models
    https://github.com/keras-team/keras/issues/12261
    https://stackoverflow.com/questions/40278868/keras-use-the-same-layer-in-different-models-share-weights
    """

    user_direct = Input(shape=(1,), name='user_direct_input')  # shape=(?,1)
    item_direct = Input(shape=(1,), name='item_direct_input')  # shape=(?,1)
    user_vertex_high_order = Input(shape=(1,), name='user_vertex_high_order_input')  # shape=(?,1)
    user_context_high_order = Input(shape=(1,), name='user_context_high_order_input')  # shape=(?,1)
    item_vertex_high_order = Input(shape=(1,), name='item_vertex_high_order_input')  # shape=(?,1)
    item_context_high_order = Input(shape=(1,), name='item_context_high_order_input')  # shape=(?,1)

    user_emb = Embedding(num_users, embedding_size, name='user_emb')
    item_emb = Embedding(num_items, embedding_size, name='item_emb')
    context_user_emb = Embedding(num_users, embedding_size, name='context_user_emb')
    context_item_emb = Embedding(num_items, embedding_size, name='context_item_emb')

    user_direct_emb = user_emb(user_direct)  # shape=(?,1,1024)
    item_direct_emb = item_emb(item_direct)  # shape=(?,1,1024)
    user_vertex_high_order_emb = user_emb(user_vertex_high_order)  # shape=(?,1,1024)
    user_context_high_order_emb = context_user_emb(user_context_high_order)  # shape=(?,1,1024)
    item_vertex_high_order_emb = item_emb(item_vertex_high_order)  # shape=(?,1,1024)
    item_context_high_order_emb = context_item_emb(item_context_high_order)  # shape=(?,1,1024)

    # Crucial to flatten an embedding vector!
    user_direct_latent = Flatten()(user_direct_emb)  # shape=(?,1024)
    item_direct_latent = Flatten()(item_direct_emb)  # shape=(?,1024)
    user_vertex_high_order_latent = Flatten()(user_vertex_high_order_emb)  # shape=(?,1024)
    user_context_high_order_latent = Flatten()(user_context_high_order_emb)  # shape=(?,1024)
    item_vertex_high_order_latent = Flatten()(item_vertex_high_order_emb)  # shape=(?,1024)
    item_context_high_order_latent = Flatten()(item_context_high_order_emb)  # shape=(?,1024)

    dots_direct = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='dots')([user_direct_latent, item_direct_latent])  # shape=(?,)
    dots_direct_flatten = Flatten()(dots_direct)  # shape=(?,1)
    dots_direct_sigmoid = tf.keras.activations.sigmoid(dots_direct_flatten)

    dots_user_user_high_order = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='dots')([user_vertex_high_order_latent,
                                                              user_context_high_order_latent])  # shape=(?,)
    dots_user_user_high_order_flatten = Flatten()(dots_user_user_high_order)  # shape=(?,1)
    dots_user_user_high_order_sigmoid = tf.keras.activations.sigmoid(dots_user_user_high_order_flatten)

    dots_item_item_high_order = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='dots')([item_vertex_high_order_latent,
                                                              item_context_high_order_latent])  # shape=(?,)
    dots_item_item_high_order_flatten = Flatten()(dots_item_item_high_order)  # shape=(?,1)
    dots_item_item_high_order_sigmoid = tf.keras.activations.sigmoid(dots_item_item_high_order_flatten)

    dots_user_item_high_order = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='dots')([user_vertex_high_order_latent,
                                                              item_context_high_order_latent])  # shape=(?,)
    dots_user_item_high_order_flatten = Flatten()(dots_user_item_high_order)  # shape=(?,1)
    dots_user_item_high_order_sigmoid = tf.keras.activations.sigmoid(dots_user_item_high_order_flatten)

    dots_item_user_high_order = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='dots')([item_vertex_high_order_latent,
                                                              user_context_high_order_latent])  # shape=(?,)
    dots_item_user_high_order_flatten = Flatten()(dots_item_user_high_order)  # shape=(?,1)
    dots_item_user_high_order_sigmoid = tf.keras.activations.sigmoid(dots_item_user_high_order_flatten)

    model_direct = Model(inputs=[user_direct, item_direct], outputs=[dots_direct_sigmoid])
    model_user_user_high_order = Model(inputs=[user_vertex_high_order, user_context_high_order],
                                       outputs=[dots_user_user_high_order_sigmoid])
    model_item_item_high_order = Model(inputs=[item_vertex_high_order, item_context_high_order],
                                       outputs=[dots_item_item_high_order_sigmoid])
    model_user_item_high_order = Model(inputs=[user_vertex_high_order, item_context_high_order],
                                       outputs=[dots_user_item_high_order_sigmoid])
    model_item_user_high_order = Model(inputs=[item_vertex_high_order, user_context_high_order],
                                       outputs=[dots_item_user_high_order_sigmoid])

    return model_direct, model_user_user_high_order, model_item_item_high_order, \
           model_user_item_high_order, model_item_user_high_order, {'user': user_emb, 'item': item_emb}


def graph_context_batch_iter(all_pairs, batch_size):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros((batch_size, 2), dtype=np.int32)
        labels = np.ones(batch_size, dtype=np.int32)
        batch[:, :] = all_pairs[batch_idx, :]
        yield batch, labels


class CSE:
    def __init__(self, num_users, num_items, embedding_size, data_path, epochs, edge_size, negative_ratio, walk_steps,
                 batch_size=1024, times=1):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.data_path = data_path
        self.epochs = epochs
        self.edge_size = edge_size
        self.negative_ratio = negative_ratio
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)
        self.walk_steps = walk_steps

        self.prepare_adjacency_list()
        self._gen_sampling_table()
        self.reset_training_config(batch_size, times)
        self.reset_model()

    def prepare_adjacency_list(self):
        self.all_pairs = np.loadtxt(self.data_path + 'user_adjacency_list.csv', dtype=np.int32, delimiter=' ')

        self.user_items_map = {}
        self.item_users_map = {}
        for i in range(len(self.all_pairs)):
            user = self.all_pairs[i][0]
            item = self.all_pairs[i][1]
            self.user_items_map.setdefault(user, [])
            self.user_items_map[user].append(item)
            self.item_users_map.setdefault(item, [])
            self.item_users_map[item].append(user)

    def reset_model(self, opt='adam', learning_rate=0.025):
        # self.model_direct, self.model_user_high_order, self.model_item_high_order, self.embedding_dict = \
        #     create_model(self.num_users, self.num_items, self.embedding_size)
        self.model_direct, self.model_user_user_high_order, self.model_item_item_high_order, \
            self.model_user_item_high_order, self.model_item_user_high_order, self.embedding_dict = \
            create_model(self.num_users, self.num_items, self.embedding_size)
        # self.model_direct.compile(optimizer=Adam(lr=learning_rate), loss={'tf_op_layer_Sigmoid': 'binary_crossentropy'},
        #                           loss_weights={'tf_op_layer_Sigmoid': 1.})
        self.model_direct.compile(opt, loss={'tf_op_layer_Sigmoid': 'binary_crossentropy'},
                                  loss_weights={'tf_op_layer_Sigmoid': 1.})
        self.model_user_user_high_order.compile(opt, loss={'tf_op_layer_Sigmoid_1': 'binary_crossentropy'},
                                           loss_weights={'tf_op_layer_Sigmoid_1': 0.05})
        self.model_item_item_high_order.compile(opt, loss={'tf_op_layer_Sigmoid_2': 'binary_crossentropy'},
                                           loss_weights={'tf_op_layer_Sigmoid_2': 0.05})
        self.model_user_item_high_order.compile(opt, loss={'tf_op_layer_Sigmoid_3': 'binary_crossentropy'},
                                                loss_weights={'tf_op_layer_Sigmoid_3': 0.05})
        self.model_item_user_high_order.compile(opt, loss={'tf_op_layer_Sigmoid_4': 'binary_crossentropy'},
                                                loss_weights={'tf_op_layer_Sigmoid_4': 0.05})
        self.batch_buffer = {'user_user': [], 'item_item': [], 'user_item': [], 'item_user': []}
        self.batch_it = self.batch_iter()
        self.batch_it_user_user_high_order = self.batch_iter_user_user_high_order()
        self.batch_it_item_item_high_order = self.batch_iter_item_item_high_order()
        self.batch_it_user_item_high_order = self.batch_iter_user_item_high_order()
        self.batch_it_item_user_high_order = self.batch_iter_item_user_high_order()

    def _gen_sampling_table(self):
        # create sampling table for user vertex
        power = 0.75
        user_degree = np.zeros(self.num_users)  # out degree

        for i in range(self.all_pairs.shape[0]):
            user_degree[self.all_pairs[i][0]] += 1.

        total_sum = sum([math.pow(user_degree[i], power)
                         for i in range(self.num_users)])
        norm_prob = [float(math.pow(user_degree[j], power)) /
                     total_sum for j in range(self.num_users)]

        self.user_accept, self.user_alias = create_alias_table(norm_prob)

        # create sampling table for item vertex
        item_degree = np.zeros(self.num_items)  # out degree

        for i in range(self.all_pairs.shape[0]):
            item_degree[self.all_pairs[i][1]] += 1.

        total_sum = sum([math.pow(item_degree[i], power)
                         for i in range(self.num_items)])
        norm_prob = [float(math.pow(item_degree[j], power)) /
                     total_sum for j in range(self.num_items)]

        self.item_accept, self.item_alias = create_alias_table(norm_prob)

    def sample_context_from_user(self, user):
        return random.choice(self.user_items_map[user])

    def sample_context_from_item(self, item):
        return random.choice(self.item_users_map[item])
    
    def batch_iter(self):
        """fit_generator
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        """
        while True:
            batch_features, batch_labels = next(graph_context_batch_iter(self.all_pairs, self.batch_size))
            yield ([batch_features[:, 0], batch_features[:, 1]], batch_labels)  # (1024,)

            # negative sample
            for _ in range(self.negative_ratio):
                negative_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                negative_batch[:, :1] = batch_features[:, :1]
                for i in range(self.batch_size):
                    negative_batch[i, 1] = alias_sample(self.item_accept, self.item_alias)
                yield ([negative_batch[:, 0], negative_batch[:, 1]], negative_labels)
            
            # walk from item
            batch_features_copy = np.zeros((self.batch_size, 2), dtype=np.int32)
            batch_features_copy[:, :] = batch_features[:, :]
            for j in range(self.walk_steps):
                if j % 2 == 0:  # (user, item->user)
                    high_order_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                    high_order_labels = np.ones(self.batch_size, dtype=np.int32)
                    for i in range(self.batch_size):
                        batch_features_copy[i, 1] = self.sample_context_from_item(batch_features_copy[i, 1])
                    high_order_batch[:, :] = batch_features_copy[:, :]
                    self.batch_buffer['user_user'].append(
                        ([high_order_batch[:, 0], high_order_batch[:, 1]], high_order_labels))
                
                    # negative sample
                    for _ in range(self.negative_ratio):
                        negative_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                        negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                        negative_batch[:, :1] = batch_features_copy[:, :1]
                        for i in range(self.batch_size):
                            negative_batch[i, 1] = alias_sample(self.user_accept, self.user_alias)
                        self.batch_buffer['user_user'].append(
                            ([negative_batch[:, 0], negative_batch[:, 1]], negative_labels))
                
                else:  # (user, user->item)
                    high_order_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                    high_order_labels = np.ones(self.batch_size, dtype=np.int32)
                    for i in range(self.batch_size):
                        batch_features_copy[i, 1] = self.sample_context_from_user(batch_features_copy[i, 1])
                    high_order_batch[:, :] = batch_features_copy[:, :]
                    self.batch_buffer['user_item'].append(
                        ([high_order_batch[:, 0], high_order_batch[:, 1]], high_order_labels))
                
                    # negative sample
                    for _ in range(self.negative_ratio):
                        negative_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                        negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                        negative_batch[:, :1] = batch_features_copy[:, :1]
                        for i in range(self.batch_size):
                            negative_batch[i, 1] = alias_sample(self.item_accept, self.item_alias)
                        self.batch_buffer['user_item'].append(
                            ([negative_batch[:, 0], negative_batch[:, 1]], negative_labels))

            # walk from user
            batch_features_copy = np.zeros((self.batch_size, 2), dtype=np.int32)
            batch_features_copy[:, 0] = batch_features[:, 1]
            batch_features_copy[:, 1] = batch_features[:, 0]
            for j in range(self.walk_steps):
                if j % 2 == 0:  # (item, user->item)
                    high_order_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                    high_order_labels = np.ones(self.batch_size, dtype=np.int32)
                    for i in range(self.batch_size):
                        batch_features_copy[i, 1] = self.sample_context_from_user(batch_features_copy[i, 1])
                    high_order_batch[:, :] = batch_features_copy[:, :]
                    self.batch_buffer['item_item'].append(
                        ([high_order_batch[:, 0], high_order_batch[:, 1]], high_order_labels))
                
                    # negative sample
                    for _ in range(self.negative_ratio):
                        negative_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                        negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                        negative_batch[:, :1] = batch_features_copy[:, :1]
                        for i in range(self.batch_size):
                            negative_batch[i, 1] = alias_sample(self.item_accept, self.item_alias)
                        self.batch_buffer['item_item'].append(
                            ([negative_batch[:, 0], negative_batch[:, 1]], negative_labels))
                
                else:  # (item, item->user)
                    high_order_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                    high_order_labels = np.ones(self.batch_size, dtype=np.int32)
                    for i in range(self.batch_size):
                        batch_features_copy[i, 1] = self.sample_context_from_item(batch_features_copy[i, 1])
                    high_order_batch[:, :] = batch_features_copy[:, :]
                    self.batch_buffer['item_user'].append(
                        ([high_order_batch[:, 0], high_order_batch[:, 1]], high_order_labels))
                
                    # negative sample
                    for _ in range(self.negative_ratio):
                        negative_batch = np.zeros((self.batch_size, 2), dtype=np.int32)
                        negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                        negative_batch[:, :1] = batch_features_copy[:, :1]
                        for i in range(self.batch_size):
                            negative_batch[i, 1] = alias_sample(self.user_accept, self.user_alias)
                        self.batch_buffer['item_user'].append(
                            ([negative_batch[:, 0], negative_batch[:, 1]], negative_labels))

    def batch_iter_user_user_high_order(self):
        while True:
            buffer = self.batch_buffer['user_user'][:]
            self.batch_buffer['user_user'].clear()
            for i in range(len(buffer)):
                yield buffer[i]

    def batch_iter_item_item_high_order(self):
        while True:
            buffer = self.batch_buffer['item_item'][:]
            self.batch_buffer['item_item'].clear()
            for i in range(len(buffer)):
                yield buffer[i]

    def batch_iter_user_item_high_order(self):
        while True:
            buffer = self.batch_buffer['user_item'][:]
            self.batch_buffer['user_item'].clear()
            for i in range(len(buffer)):
                yield buffer[i]

    def batch_iter_item_user_high_order(self):
        while True:
            buffer = self.batch_buffer['item_user'][:]
            self.batch_buffer['item_user'].clear()
            for i in range(len(buffer)):
                yield buffer[i]

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        # self.reset_training_config(batch_size, times)
        for i in range(self.epochs - initial_epoch):
            with elapsed_timer("-- {0}s - %s" % ("train model_direct",)):
                self.model_direct.fit_generator(self.batch_it, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                                                steps_per_epoch=self.steps_per_epoch, verbose=verbose)
            with elapsed_timer("-- {0}s - %s" % ("train model_user_user_high_order",)):
                self.model_user_user_high_order.fit_generator(
                    self.batch_it_user_user_high_order, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                    steps_per_epoch=self.steps_per_epoch * ((self.walk_steps+1)//2), verbose=verbose)
            with elapsed_timer("-- {0}s - %s" % ("train model_item_item_high_order",)):
                self.model_item_item_high_order.fit_generator(
                    self.batch_it_item_item_high_order, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                    steps_per_epoch=self.steps_per_epoch * ((self.walk_steps+1)//2), verbose=verbose)
            with elapsed_timer("-- {0}s - %s" % ("train model_user_item_high_order",)):
                self.model_user_item_high_order.fit_generator(
                    self.batch_it_user_item_high_order, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                    steps_per_epoch=self.steps_per_epoch * (self.walk_steps//2), verbose=verbose)
            with elapsed_timer("-- {0}s - %s" % ("train model_item_user_high_order",)):
                self.model_item_user_high_order.fit_generator(
                    self.batch_it_item_user_high_order, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                    steps_per_epoch=self.steps_per_epoch * (self.walk_steps//2), verbose=verbose)

    def get_embeddings(self,):
        user_embeddings = self.embedding_dict['user'].get_weights()[0]
        item_embeddings = self.embedding_dict['item'].get_weights()[0]

        return user_embeddings, item_embeddings

    # def save(self):
    #     self.model.save(os.path.join(self.data_path, "CSE.h5"), overwrite=True)


def output_embeddings(user_embeddings, item_embeddings):
    with open("../../../data/amazon/embedding/CSE_user.embed", "w") as file:
        num_users, embedding_size = user_embeddings.shape
        file.write(str(num_users) + " " + str(embedding_size) + "\n")
        for user in range(num_users):
            file.write(str(user) + " " + " ".join([str(e) for e in user_embeddings[user]]) + "\n")
    with open("../../../data/amazon/embedding/CSE_item.embed", "w") as file:
        num_items, embedding_size = item_embeddings.shape
        file.write(str(num_items) + " " + str(embedding_size) + "\n")
        for item in range(num_items):
            file.write(str(item) + " " + " ".join([str(e) for e in item_embeddings[item]]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--num_users", type=int, default=2558)
    parser.add_argument("--num_items", type=int, default=4091)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default='../../../data/amazon/')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--edge_size", type=int, default=5126)
    parser.add_argument("--negative_ratio", type=int, default=5)
    parser.add_argument("--walk_steps", type=int, default=5)
    args = parser.parse_args()

    model = CSE(args.num_users, args.num_items, args.embedding_size, args.data_path, args.epochs, args.edge_size,
                args.negative_ratio, args.walk_steps)
    model.train()
    user_embeddings, item_embeddings = model.get_embeddings()
    output_embeddings(user_embeddings, item_embeddings)
    # model.save()
