#
#       Author:                 Michail Tamvakeras
#       Postgraduate course:    MSc in Intelligent Information Systems (MIIS)
#       University:             University of the Aegean
#       Department:             Information and Communication Systems Engineering
#
#       MSc Thesis:             Facial emotion recognition using deep learning
#

import datetime
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Just disables the warning, doesn't enable AVX/FMA

from CNNetwork import CNNetwork
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import tensorflow as tf
from VisualizationTools import plot_loss_history, plot_accuracy_history, plot_learning_rate_history, plot_confusion_matrix, setup_plot


class FerVGG16(CNNetwork):
    '''
    This class is derived from the base class CNNetwork, which implements a custom
    training and testing behaviour of the VGG16 model.
    '''
    _saved_model_dir = './logs/vgg16/'

    def __init__(self, mini_batch_size=10, initial_learning_rate=0.01, max_epochs=10,
        net_name=None, image_size=48, image_type='rgb'):

        assert net_name is not None

        # Calls the base class first
        CNNetwork.__init__(self, mini_batch_size, initial_learning_rate, max_epochs, net_name, image_size, image_type)
        return


    def build(self, num_emotions):
        """
        This method builds the adapted VGG16 model
        INPUT:
        - num_emotions: The amount of the emotion classes:
          0=Angry/Disgust, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
        """
        assert num_emotions == 6
        self._num_emotions = num_emotions   # Stores the number of the emotion classes

        # Setups the model
        vgg16 = VGG16(
            input_shape=[self._image_size, self._image_size, 3],   # The image size the network requires
            weights='imagenet', # The pre-trained weights from the ImageNet
            pooling='max',      # Uses a max-pooling layer as last pooling layer
            include_top=False   # Excludes the last layer of the network
        )

        for layer in vgg16.layers[:3]:  # Freezes all layers until the penultimate convolutional layer
            layer.trainable = False
        for layer in vgg16.layers[3:]:  # Unfreezes all layers until the penultimate convolutional layer
            layer.trainable = True

        # Adapts the last layers to classify the six emotion classes
        batchnorm = BatchNormalization()(vgg16.output)
        dropout_layer = Dropout(0.2)(batchnorm)  # Disables randomly 20% of the neurons
        predictions = Dense(units=num_emotions, activation='softmax', name='predictions')(dropout_layer)

        self._learning_rate = self._initial_learning_rate   # Stores the initial learning rate

        # Configures the learning process
        model = Model(inputs=vgg16.input, outputs=predictions)
        model.compile(
            optimizer=Adam(lr=self._learning_rate),  # The optimizer and the learning rate to use
            loss='categorical_crossentropy',    # The loss function
            metrics=['accuracy']                # The metrics to calculate
        )
        print(model.summary())  # Prints the model structure

        # Checks if there is a saved model available, if not the new one gets stored
        train_save_path = self._saved_model_dir + "train/"
        files = [file for file in os.listdir(train_save_path)]
        if len(files) is 0:
            model.save(train_save_path + "init.h5")
        return


    def create_generator(self, batch_size, mini_batch_type='training', aug=None):
        """
        This method creates a generator to reads the mini-batch data from disk
        and returns them to be feed into the network
        INPUT:
        - batch_size: The batch size to of the images
        - mini_batch_type: Either training or validation
        - aug: If the data should be augmented or not. Has to be None for validation
        IMPORTANT: The data reading process has to be unlimited and only specified by the epoch number
        """
        index = 0
        while True:
            # Reads the images and labels from the hdf5 file in successive order
            images, labels, index = self.get_successive_batch(index, mini_batch_type)

            # Preprocess the images if necessary
            if images[0].shape[1] != self._image_size or images[0].shape[2] != self._image_size:
                images, labels = self.preprocess_images(images, labels)

            one_hot_labels = to_categorical(labels, self._num_emotions)
            # If augmentation is desired
            if aug is not None:
                (images, one_hot_labels) = next(aug.flow(images, one_hot_labels, batch_size=batch_size))
            if index is -1: # Resets the index to the start again and checks for learning rate decay
                index = 0

            yield(images, one_hot_labels)   # Returns the images and labels to the calling method


    # CUSTOM CALLBACK-METHOD
    def rate_decay(self, epoch):
        """
        This callback method calculates the new learning rate to apply learning rate decay
        INPUT:
        - epoch: The current epoch
        """
        self._learning_rate = self._initial_learning_rate / (
                1 + self._decay_rate * epoch)  # Decays the learning rate
        return np.float32(self._learning_rate)


    def training(self, augmentation=True, early_stopping=True, decay_rate=-1):
        """
        This method trains the network
        INPUT:
        - augmentation: Flag which determines if the training set gets augmented or not
        - early_stopping: Flag to apply early stopping
        - decay_rate: The learning decay rate, if -1 not learning rate decay gets applied
        """

        max_train, max_valid, _ = self.get_max_dataset_values() # Gets the amount of training examples
        train_save_path = self._saved_model_dir + "train/"
        start = datetime.datetime.now()

        # Checks if augmentation is desired
        if augmentation:
            # Creates an ImageDataGenerator which applies augmentation
            aug = ImageDataGenerator(rotation_range=25, zoom_range=0.17, width_shift_range=0.2,
                height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
        else:
            aug = None

        # Creates the training and validation generator to read the data
        trainGen = self.create_generator(self._mini_batch_size, 'training', aug=aug)
        validGen = self.create_generator(self._mini_batch_size, 'validation', aug=None)

        # Loads a saved model
        files = [file for file in os.listdir(train_save_path)]
        model = load_model(train_save_path + files[0])

        # Setups the callback functions
        callback_list = []
        checkpoint = ModelCheckpoint(train_save_path+'epoch-{epoch:02d}-val_acc_{val_acc:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # Checks if learning rate decay is desired
        if decay_rate is not -1:
            self._decay_rate = decay_rate
            learning_rate_decay = LearningRateScheduler(self.rate_decay)
            callback_list.append(learning_rate_decay)
        # Checks if early stopping is desired
        if early_stopping:
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)
            callback_list.append(early_stop)
        callback_list.append(checkpoint)

        # Creates a custom callback function to get the data for the confusion matrix
        predictions_cbk = self.PredictionHistory()
        fetches = [tf.assign(predictions_cbk._true, model.targets[0], validate_shape=False),
                   tf.assign(predictions_cbk._pred, model.outputs[0], validate_shape=False)]
        model._function_kwargs = {'fetches': fetches}
        callback_list.append(predictions_cbk)

        # Starts training the network
        history = model.fit_generator(
            trainGen,
            steps_per_epoch=max_train // self._mini_batch_size,
            validation_data=validGen,
            validation_steps=max_valid // self._mini_batch_size,
            epochs=self._max_epochs,
            callbacks=callback_list)

        # PLOTS THE RESULTS
        setup_plot(font_size=22, show_grid=False)
        plot_loss_history(history.history['loss'], history.history['val_loss'])     # Plots the loss history
        plot_accuracy_history(history.history['acc'], history.history['val_acc'])   # Plots the accuracy history
        plot_learning_rate_history(history.history['lr'])   # Plots the learning rate history

        # Plots the confusion matrix
        class_labels = ["Angry/Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        plot_names =["Angry,\nDisgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        target_list = []; prediction_list = []
        # Create the target and prediction lists for the confusion matrix
        for item in predictions_cbk._targets:
            for val in item:
                target_list.append(self.code_to_emotion(val))
        for item in predictions_cbk._predictions:
            for val in item:
                prediction_list.append(self.code_to_emotion(val))
        invalid_labels = [n for n in class_labels if n not in target_list and n not in prediction_list]  # Reads the invalid values
        plot_confusion_matrix(target_list, prediction_list, class_labels, plot_names, invalid_labels)

        # Prints the result after one epoch
        end = datetime.datetime.now()
        print("Training ended")
        print("Overall duration: " + str(end - start))
        return


    def predict(self, image, model):
        """
        This method does the prediction of unknown images
        INPUT:
        - image: The image to predict
        - model: Model to test
        RETURN:
        - The accurracy of the classification step
        """
        if image.shape[0] != self._image_size or image.shape[1] != self._image_size:
            image = self.preprocess_image(image)

        # Executes the prediction
        predictions = model.predict(image)
        percentages = predictions * 100.0
        return percentages[0]


    # TODO: Implementation of the testing process for the VGG16 network
    def testing(self, num_emotions, model):
        """
        This method tests the network
        INPUT:
        - num_emotions: The amount of the emotion classes:
          0=Angry/Disgust, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
        """
        assert num_emotions == 6
        self._num_emotions = num_emotions  # Stores the number of the emotion classes

        print("Testing starts...")

        _, _, max_test = self.get_max_dataset_values()  # Gets the amount of testing examples
        train_save_path = self._saved_model_dir + "train/"
        start = datetime.datetime.now()

        # Creates the testing generator to read the data
        testGen = self.create_generator(self._mini_batch_size, 'testing')

        # Loads a saved model
        files = [file for file in os.listdir(train_save_path)]
        model = load_model(train_save_path + files[0])

        # Executes the testing process
        predictions = model.evaluate_generator(
            testGen,
            steps=max_test // self._mini_batch_size,
            verbose=1,
            workers=1)

        end = datetime.datetime.now()
        print("Testing has ended")
        print("Overall duration: " + str(end - start))
        return predictions[1] * 100.0


    def code_to_emotion(self, code):
        """
        This method converts the emotion number to the appropriate emotion name
        INPUT:
        - code: The integer code of the emotion
        RETURN:
        - The string name of the respective emotion code, otherwise None
        """
        emotion_names = {0: "Angry/Disgust",
                         1: "Fear",
                         2: "Happy",
                         3: "Sad",
                         4: "Surprise",
                         5: "Neutral"}

        if code >=0 and code <=5:
            return emotion_names.get(code)
        return None