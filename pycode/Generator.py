import tensorflow as tf
import numpy as np
from tensorflow import keras
from math import floor

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, batch_size=128, dim=(128, 128), n_channels = 25, shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y, w = self.__data_generation(indexes)
        return X, y, w

    def on_epoch_end(self):
        'Updates indexes after each epoch. It is also executed at the very beginning'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels) El asterisco expande la variable (32, *(128, 128)) = (32, 128, 128)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) #
        y = np.empty((self.batch_size, *self.dim), dtype=int)
        w = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, index in enumerate(index_list):
            # Store img, label and weight
            X[i], y[i], w[i] = self.dataset[index]

        return X, y, w
