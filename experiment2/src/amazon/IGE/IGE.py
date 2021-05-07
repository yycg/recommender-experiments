import numpy as np
from tensorflow.python.keras.layers import Embedding, Input, Flatten, Concatenate, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
import argparse

def create_model(num_users, num_items, embedding_size):
    u = Input(shape=(1,), name='user_input')
    v_i = Input(shape=(1,), name='item_input')

    user_emb = Embedding(num_users, embedding_size, name='user_emb')
    item_emb = Embedding(num_items, embedding_size, name='item_emb')

    u_emb = user_emb(u)
    v_i_emb = item_emb(v_i)

    # Crucial to flatten an embedding vector!
    # https://stackoverflow.com/questions/48855804/what-does-flatten-do-in-sequential-model-in-keras
    user_latent = Flatten()(u_emb)
    item_latent = Flatten()(v_i_emb)

    # The 0-th layer is the concatenation of embedding layers
    # https://keras.io/api/layers/merging_layers/concatenate/
    vector = Concatenate()([user_latent, item_latent])

    # https://keras.io/api/layers/core_layers/dense/
    # https://keras.io/api/layers/activations/
    hidden = Dense(embedding_size, activation='relu', name="hidden")(vector)

    softmax = Dense(num_items, activation='softmax', name='softmax')(hidden)

    # https://keras.io/api/models/model/
    model = Model(inputs=[u, v_i], outputs=softmax)

    return model, {'user': user_emb, 'item': item_emb}

def graph_context_batch_iter(s_u, batch_size):
    while True:
        start_idx = np.random.randint(0, len(s_u) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros((batch_size, 2), dtype=np.int32)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = s_u[batch_idx, :2]
        labels[:, 0] = s_u[batch_idx, 2]
        yield batch, labels

class IGE:
    def __init__(self, num_users, num_items, embedding_size, data_path, epochs, edge_size, negative_ratio,
                 batch_size=1024, times=1):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.data_path = data_path
        self.epochs = epochs
        self.edge_size = edge_size
        self.negative_ratio = negative_ratio
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)

        self.reset_training_config(batch_size, times)
        self.reset_model()

    def reset_model(self, opt='adam'):
        self.model, self.embedding_dict = create_model(self.num_users, self.num_items, self.embedding_size)
        self.model.compile(opt, 'categorical_crossentropy')
        self.batch_it = self.batch_iter()

    def batch_iter(self):
        """fit_generator
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        """
        while True:
            all_pairs = np.loadtxt(self.data_path + 's_u.csv', dtype=np.int32, delimiter=' ')
            batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, self.batch_size))
            yield ([batch_features[:, 0], batch_features[:, 1]], to_categorical(batch_labels, num_classes=self.num_items))

            # TODO: negative

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        # self.reset_training_config(batch_size, times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch,
                                        steps_per_epoch=self.steps_per_epoch,
                                        verbose=verbose)
        return hist

    def get_embeddings(self,):
        user_embeddings = self.embedding_dict['user'].get_weights()[0]
        item_embeddings = self.embedding_dict['item'].get_weights()[0]

        return user_embeddings, item_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--num_users", type=int, default=2558)
    parser.add_argument("--num_items", type=int, default=4091)
    parser.add_argument("--embedding_size", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default='../../../data/amazon/')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--edge_size", type=int, default=5434)
    parser.add_argument("--negative_ratio", type=int, default=5)
    args = parser.parse_args()

    model = IGE(args.num_users, args.num_items, args.embedding_size, args.data_path, args.epochs, args.edge_size,
                args.negative_ratio)
    model.train()
