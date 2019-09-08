#
#       Project:             Facial emotion recognition using deep learning
#       Developer:           Michail Tamvakeras
#

import cv2
import numpy as np
import h5py

from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf


class CNNetwork:
    '''
    This class is the base class of the CNN's, which defines the common properties and operations.
    '''

    _logs = "./logs/"   # Checkpoint and log directories
    fer_img_dataset_file = "./hdf5/fer2013_dataset.hdf5"  # Path to the datasets


    def __init__(self, mini_batch_size, learning_rate, max_epochs, net_name, image_size, image_type):

        self._sess = None
        self._mini_batch_size = mini_batch_size         # Mini-batch size
        self._initial_learning_rate = learning_rate     # Initial learning rate
        self._max_epochs = max_epochs                   # Number of epochs
        self._net_name = net_name                       # Name of the network type
        self._image_size = image_size                   # Image size of the input
        self._image_type = image_type                   # Type of the image (RGB or Gray)
        return


    # ABSTRACT METHODS, THEY HAVE TO BE IMPLEMENTED IN THE SUBCLASSES
    def build(self, *args, **kwargs):
        raise NotImplementedError("Subclass has to specify build method!!!.")

    def training(self):
        raise NotImplementedError("Subclass has to specify the training method!!!.")

    def predict(self, data):
        raise NotImplementedError("Subclass has to specify the prediction method!!!.")

    # TODO: Abstract test method
    def testing(self):
        raise NotImplementedError("Subclass has to specify the test method!!!.")

    def get_name(self):
        return self._net_name


    def get_max_dataset_values(self):
        """
        This method returns the amount of each dataset (training, validation and test)
        RETURN:
        - max_train: Maximum number of training examples
        - max_valid: Maximum number of validation examples
        - max_test: Maximum number of test examples
        """
        dataset_file = self.fer_img_dataset_file
        with h5py.File(dataset_file, mode='r') as hdf5_file:
            max_train = hdf5_file['training_img'].shape[0]
            max_valid = hdf5_file['validation_img'].shape[0]
            max_test = hdf5_file['testing_img'].shape[0]
        return max_train, max_valid, max_test


    def get_successive_batch(self, current_index, mini_batch_type='training'):
        """
        This method loads the next mini-batch of the given mini-batch size and type in successive order.
        INPUT:
        - current_index: The current example index given
        - mini_batch_type: The type of the dataset (training, validation or testing) to access
        RETURN:
        - images: The mini-batch containing the images in successive order
        - labels: The mini-batch containing the labels in successive order
        - index: The index of the current example of the dataset
        """
        dataset_file = self.fer_img_dataset_file

        # Gets the next mini-batch of the image dataset
        with h5py.File(dataset_file, mode='r') as hdf5_file:

            # Loads the mini-batch for training
            dataset_img_type = mini_batch_type + "_img"
            dataset_label_type = mini_batch_type + "_label"
            max_examples = hdf5_file[dataset_img_type].shape[0]

            index = None
            images = None
            labels = None
            diff = max_examples - current_index

            # Last mini-batch
            if diff <= self._mini_batch_size:
                images = hdf5_file[dataset_img_type][current_index:max_examples]
                labels = hdf5_file[dataset_label_type][current_index:max_examples]
                index = -1
            # There are still mini-batches to proceed
            elif diff > self._mini_batch_size:
                images = hdf5_file[dataset_img_type][current_index:current_index + self._mini_batch_size]
                labels = hdf5_file[dataset_label_type][current_index:current_index + self._mini_batch_size]
                index = current_index + self._mini_batch_size
            return images, labels, index


    def preprocess_images(self, img_batch=None, label_batch=None):
        """
        This method pre-process the mini-batch to get fit into the network
        INPUT:
        - img_batch: Image batch to process
        - label_batch: The labels to process
        RETURN:
        - preproc_image_batch: The pre-processed images for the network
        - preproc_label_batch: The pre-processed labels for the network
        """
        assert img_batch is not None
        assert label_batch is not None

        preproc_label_batch = np.zeros((label_batch.shape[0], 1), dtype = np.uint8)
        preproc_image_batch = None

        # Checks if the given image is a grayscale or RGB image
        # GRAY:
        if self._image_type is 'gray':
            # Creates a list of the appropriate size and appends the resized images and the labels
            preproc_image_batch = np.zeros((img_batch.shape[0], self._image_size, self._image_size, 1), dtype=np.float32)
            for i, (img, lab) in enumerate(zip(img_batch, label_batch)):
                img = cv2.resize(img, dsize=(self._image_size, self._image_size))   # Resizes the image
                img = np.float32(img/255)   # Normalizes the image in the range [0, 1]
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                preproc_image_batch[i, :, :, :] = img  # Stores the image
                preproc_label_batch[i, :] = lab
        # RGB:
        elif self._image_type is 'rgb':
            # Creates a list of the appropriate size and appends the resized images
            preproc_image_batch = np.zeros((img_batch.shape[0], self._image_size, self._image_size, 3), dtype=np.float32)
            for i, (img, lab) in enumerate(zip(img_batch, label_batch)):
                img = cv2.resize(img, dsize=(self._image_size, self._image_size))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = np.float32(img/255)
                preproc_image_batch[i, :, :, :] = img
                preproc_label_batch[i, :] = lab
        return preproc_image_batch, preproc_label_batch


    def preprocess_image(self, image=None):
        """
        This method pre-process an image, to fit into the network.
        INPUT:
        - image: Image to process
        RETURN:
        - new_image: The pre-processed image for the network
        """
        assert image is not None

        new_image = None
        proc_img = None
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Filter to apply to sharpen the images
        proc_img = cv2.resize(image, dsize=(self._image_size, self._image_size))
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2RGB)
        proc_img = cv2.filter2D(proc_img, -1, filter)  # Apply filter to sharpen the image
        proc_img = np.float32(proc_img/255)
        new_image = np.zeros((1, self._image_size, self._image_size, 3), dtype=np.float32)
        new_image[0:,:,:,:] = proc_img
        return new_image


    # CUSTOM CALLBACK CLASS
    class PredictionHistory(Callback):
        # Based on: https://stackoverflow.com/questions/47079111/create-keras-callback-to-save-model-predictions-and-targets-for-each-batch-durin

        def __init__(self):
            Callback.__init__(self)
            self._targets = []
            self._predictions = []
            self._true = tf.Variable(0., validate_shape=False)
            self._pred = tf.Variable(0., validate_shape=False)

        def on_batch_end(self, batch, logs=None):
            self._targets.append(np.argmax(K.eval(self._true), axis=1))
            self._predictions.append(np.argmax(K.eval(self._pred), axis=1))

