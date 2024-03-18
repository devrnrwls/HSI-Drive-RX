import numpy as np
import tensorflow as tf
import cv2
import os

class HSIDriveDataset():
    """
    A class used to load image/label pairs from a set of directories
    Attributes
    ----------
    dirs : str, list
        Paths for each parent directory
    paths : list
        Paths to every subdirectory within self.dirs that contain valid image/maks pairs.
    """
    def __init__(self, dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, inplicit_norm=False, usePatches=True):
        self.dirs = dirs
        self.inplicit_norm = inplicit_norm
        self.normMaxValue = np.array(normMaxValue)
        self.normMinValue = np.array(normMinValue)

        self.data_path = [os.path.join(os.path.join(self.dirs, data_folder), i) for i in os.listdir(os.path.join(self.dirs, data_folder))]
        self.label_path = [os.path.join(os.path.join(self.dirs, label_folder), i) for i in os.listdir(os.path.join(self.dirs, label_folder))]

        if len(self.data_path) != len(self.label_path):
            raise Exception("El número de cubos no es igual al número de etiquetas.")

        self.data_path.sort()
        self.label_path.sort()
        self.weights = np.array(weights)

        if usePatches:
            print('HSI-Drive dataset with {} patches'.format(len(self.data_path)))
        else:
            print('HSI-Drive dataset with {} images'.format(len(self.data_path)))

    def __len__(self):
        return len(self.data_path)


    def __getitem__(self, index):

        if self.inplicit_norm:
            img = np.load(self.data_path[index])
        else:
            img = np.array(self.normalizeImageNpy(self.data_path[index], "other", self.normMaxValue, self.normMinValue))
        label = np.array(cv2.imread(self.label_path[index], 1)[:, : , 0])
        sample_weights = tf.gather(self.weights, indices=tf.cast(label, tf.int32))

        try:
            return img, label, sample_weights
        except BaseException:
            print('Could not read:', self.data_path[index])

    def getitemname(self, index):
        return self.data_path[index], self.label_path[index]

    def normalizeImageNpy(self, path, method, maximo, minimo):
        mat = np.load(path).astype(np.float32)
        num_channels = maximo.shape[0]
        if method == 'removeMean':
            for i in range(num_channels):
                mat[:, :, i] = mat[:, :, i] - np.mean(mat[:, :, i])
                mat[:, :, i] = 2*((mat[:, :, i] - minimo[i])/(maximo[i] - minimo[i])) - 1

        else:
            for i in range(num_channels):
                mat[:, :, i] = 2*( (mat[:, :, i] - minimo[i]) / (maximo[i] - minimo[i]) ) - 1

        return mat

