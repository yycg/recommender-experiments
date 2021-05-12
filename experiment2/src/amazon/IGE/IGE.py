import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Input, Flatten, Concatenate, Dense, Lambda, Softmax
from tensorflow.python.keras.models import Model
import argparse
import random
import os


class IGE:
    def __init__(self, num_users, num_items, embedding_size, data_path, epochs, sample_size, sample_size_item,
                 negative_ratio, batch_size=1024, times=1):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.data_path = data_path
        self.epochs = epochs
        self.sample_size = sample_size
        self.sample_size_item = sample_size_item
        self.negative_ratio = negative_ratio
        self.samples_per_epoch = self.sample_size * (1 + negative_ratio)
        self.samples_per_epoch_item = self.sample_size_item * (1 + negative_ratio)

        self.reset_training_config(batch_size, times)
        self.reset_model()

    def create_model(self, num_users, num_items, embedding_size):
        """Reference: Tensorflow Word2Vec tutorial
        https://www.tensorflow.org/tutorials/text/word2vec
        """

        user_vertex = Input(shape=(1,), name='user_vertex')  # shape=(?,1)
        user_context = Input(shape=(1,), name='user_context')  # shape=(?,1)
        item_vertex = Input(shape=(1,), name='item_vertex')  # shape=(?,1)
        item_context = Input(shape=(1,), name='item_context')  # shape=(?,1)

        user_emb = Embedding(num_users, embedding_size, name='user_emb')
        item_emb = Embedding(num_items, embedding_size, name='item_emb')
        context_user_emb = Embedding(num_users, embedding_size * 2, name='context_user_emb')
        context_item_emb = Embedding(num_items, embedding_size * 2, name='context_item_emb')

        user_vertex_emb = user_emb(user_vertex)  # shape=(?,1,1024)
        item_vertex_emb = item_emb(item_vertex)  # shape=(?,1,1024)
        user_context_emb = context_user_emb(user_context)  # shape=(?,1,2048)
        item_context_emb = context_item_emb(item_context)  # shape=(?,1,2048)

        # Crucial to flatten an embedding vector!
        user_vertex_latent = Flatten()(user_vertex_emb)  # shape=(?,1024)
        item_vertex_latent = Flatten()(item_vertex_emb)  # shape=(?,1024)
        user_context_latent = Flatten()(user_context_emb)  # shape=(?,2048)
        item_context_latent = Flatten()(item_context_emb)  # shape=(?,2048)

        vector = Concatenate()([user_vertex_latent, item_vertex_latent])  # shape=(?,2048)
        dots = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keep_dims=False), name='dots')([vector, item_context_latent])  # shape=(?,)
        vector = Flatten()(dots)  # shape=(?,1)
        sigmoid = tf.keras.activations.sigmoid(vector)

        vector_item = Concatenate()([item_vertex_latent, user_vertex_latent])  # shape=(?,2048)
        dots_item = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keep_dims=False), name='dots')([vector_item, user_context_latent])  # shape=(?,)
        vector_item = Flatten()(dots_item)  # shape=(?,1)
        sigmoid_item = tf.keras.activations.sigmoid(vector_item)

        model = Model(inputs=[user_vertex, item_vertex, item_context], outputs=sigmoid)
        model_item = Model(inputs=[item_vertex, user_vertex, user_context], outputs=sigmoid_item)

        return model, model_item, {'user': user_emb, 'item': item_emb}

    def graph_context_batch_iter(self, s_u, batch_size):
        while True:
            start_idx = np.random.randint(0, len(s_u) - batch_size)
            batch_idx = np.array(range(start_idx, start_idx + batch_size))
            batch_idx = np.random.permutation(batch_idx)
            batch = np.zeros((batch_size, 3), dtype=np.int32)
            labels = np.ones(batch_size, dtype=np.int32)
            batch[:, :] = s_u[batch_idx, :]
            yield batch, labels

    def reset_model(self, opt='adam'):
        self.model, self.model_item, self.embedding_dict = \
            self.create_model(self.num_users, self.num_items, self.embedding_size)
        self.model.compile(opt, loss='binary_crossentropy')
        self.model_item.compile(opt, loss='binary_crossentropy')
        self.batch_it = self.batch_iter()
        self.batch_it_item = self.batch_iter_item()

    def batch_iter(self):
        """fit_generator
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        """
        while True:
            all_pairs = np.loadtxt(self.data_path + 's_u.csv', dtype=np.int32, delimiter=' ')
            batch_features, batch_labels = next(self.graph_context_batch_iter(all_pairs, self.batch_size))
            yield ([batch_features[:, 0], batch_features[:, 1], batch_features[:, 2]],
                   batch_labels)  # (1024,)

            # negative sample
            for _ in range(self.negative_ratio):
                negative_batch = np.zeros((self.batch_size, 3), dtype=np.int32)
                negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                negative_batch[:, :2] = batch_features[:, :2]
                for i in range(self.batch_size):
                    negative_batch[i, 2] = self._negative_sample(true_class=batch_features[i, 2],
                                                                 num_classes=self.num_items)
                yield ([negative_batch[:, 0], negative_batch[:, 1], negative_batch[:, 2]],
                       negative_labels)

    def batch_iter_item(self):
        while True:
            all_pairs = np.loadtxt(self.data_path + 's_v.csv', dtype=np.int32, delimiter=' ')
            batch_features, batch_labels = next(self.graph_context_batch_iter(all_pairs, self.batch_size))
            yield ([batch_features[:, 0], batch_features[:, 1], batch_features[:, 2]],
                   batch_labels)  # (1024,)

            # negative sample
            for _ in range(self.negative_ratio):
                negative_batch = np.zeros((self.batch_size, 3), dtype=np.int32)
                negative_labels = np.zeros(self.batch_size, dtype=np.int32)
                negative_batch[:, :2] = batch_features[:, :2]
                for i in range(self.batch_size):
                    negative_batch[i, 2] = self._negative_sample(true_class=batch_features[i, 2],
                                                                 num_classes=self.num_users)
                yield ([negative_batch[:, 0], negative_batch[:, 1], negative_batch[:, 2]],
                       negative_labels)

    def _negative_sample(self, true_class, num_classes):
        while True:
            sample = random.randint(0, num_classes-1)
            if not sample == true_class:
                return sample

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times
        self.steps_per_epoch_item = (
             (self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        # self.reset_training_config(batch_size, times)
        for i in range(self.epochs - initial_epoch):
            self.model.fit_generator(self.batch_it, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                                     steps_per_epoch=self.steps_per_epoch, verbose=verbose)
            self.model_item.fit_generator(self.batch_it_item, epochs=initial_epoch+i+1, initial_epoch=initial_epoch+i,
                                          steps_per_epoch=self.steps_per_epoch_item, verbose=verbose)

    def get_embeddings(self,):
        user_embeddings = self.embedding_dict['user'].get_weights()[0]
        item_embeddings = self.embedding_dict['item'].get_weights()[0]

        return user_embeddings, item_embeddings

    def save(self):
        self.model.save(os.path.join(self.data_path, "IGE.h5"), overwrite=True)


def output_embeddings(user_embeddings, item_embeddings):
    with open("../../../data/amazon/embedding/IGE_user.embed", "w") as file:
        num_users, embedding_size = user_embeddings.shape
        file.write(str(num_users) + " " + str(embedding_size) + "\n")
        for user in range(num_users):
            file.write(str(user) + " " + " ".join([str(e) for e in user_embeddings[user]]) + "\n")
    with open("../../../data/amazon/embedding/IGE_item.embed", "w") as file:
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=12640)
    parser.add_argument("--sample_size_item", type=int, default=10566)
    parser.add_argument("--negative_ratio", type=int, default=5)
    args = parser.parse_args()

    model = IGE(args.num_users, args.num_items, args.embedding_size, args.data_path, args.epochs, args.sample_size,
                args.sample_size_item, args.negative_ratio)
    model.train()
    user_embeddings, item_embeddings = model.get_embeddings()
    output_embeddings(user_embeddings, item_embeddings)
    # model.save()