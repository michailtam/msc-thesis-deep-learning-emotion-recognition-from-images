#
#       Project:             Facial emotion recognition using deep learning
#       Developer:           Michail Tamvakeras
#

import numpy as np
import pandas as pd
import os
import datetime
import cv2
import h5py


class FER2013DatasetCreation():

    _dataset_path = './datasets/'   # The path of the FER2013 dataset
    _num_emotions = 6               # The number of the emotion classes to use
    _image_size = 48                # The original FER2013 image properties (48x48 grayscale images)


    def __init__(self, csv_file=None):

        assert csv_file is not None

        self._train_X = []
        self._train_Y = []
        self._valid_X = []
        self._valid_Y = []
        self._test_X = []
        self._test_Y = []

        # Checks if the log folder and subfolders exists
        if os.path.exists(self._dataset_path) == False:
            os.makedirs(self._dataset_path)

        self.import_csv(csv_file)   # Reads the raw csv file content
        self.create_dataset()      # Creates the dataset
        return


    def import_csv(self, file):
        """
        This method reads the FER2013 dataset from the provided path and creates the test, validation and training set.
        The FER2013 dataset is one of the most used dataset in Kaggle facial expression recognitions competions:
        (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data),
        and is divided into 60% training, 20% validation (PrivateTest) and 20% test (PublicTest) examples.
        The emotion classes are: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        INPUT:
        - file: A file containing the dataset in csv format
        """

        # Reads the csv file and extracts the emotion and pixels column
        dataframe = pd.read_csv(file, delimiter=',', dtype='a')
        classes = np.array(dataframe['emotion'], np.float)

        # Creates the image matrices
        data = np.array(dataframe['pixels'])
        images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in data])
        del data
        num_shape = int(np.sqrt(images.shape[-1]))
        images.shape = (images.shape[0], num_shape, num_shape)

        # Saves the train/test text
        usage = np.array(dataframe['Usage'])

        # Creates the training set
        train_index = np.where(usage == 'PublicTest')
        train_index = train_index[0][0]
        self._train_X = images[:train_index]
        self._train_Y = classes[:train_index]
        self._train_Y = self._train_Y.reshape((self._train_Y.size, 1))

        # Creates the validation set
        valid_index = np.where(usage == 'PrivateTest')
        valid_index = valid_index[0][0]
        self._valid_X = images[train_index:valid_index]
        self._valid_Y = classes[train_index:valid_index]
        self._valid_Y = self._valid_Y.reshape((self._valid_Y.size, 1))

        # Creates the test set
        self._test_X = images[valid_index:]
        self._test_Y = classes[valid_index:]
        self._test_Y = self._test_Y.reshape((self._test_Y.size, 1))

        # Displays the train/test set amount and the respective shapes
        print("The FER2013 dataset consists of:")
        print("----------------------------------------------------------------------------------------")
        print("[Training]\t\tImages:\t" + str(len(self._train_X)) + "\t" + str(self._train_X.shape)
              + "\t\tLabels:\t" + str(self._train_Y.size) + "\t" + str(self._train_Y.shape)
              + "\n[Validation]\tImages:\t" + str(len(self._valid_X)) + "\t" + str(self._valid_X.shape)
              + "\t\tLabels:\t" + str(self._valid_Y.size) + "\t" + str(self._valid_Y.shape)
              + "\n[Test]\t\t\tImages:\t" + str(len(self._test_X)) + "\t" + str(self._test_X.shape)
              + "\t\tLabels:\t" + str(self._test_Y.size) + "\t" + str(self._test_Y.shape))
        print("[Emotions]\t\t0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral\n")
        return


    def create_dataset(self):
        """
        This method creates the fer2013 dataset, the files will be stored as HDF5 file.
        IMPORTANT: The labels angry and fear will be merged into one class, because their emotions are similar and each
        of them contains less than 1000 images: So the followigng classes will be created:
        0=Angry/Disgust, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
        """

        # Creates the HDF5 file
        hdf5_path = "./hdf5/fer2013_dataset.hdf5"
        with h5py.File(hdf5_path, mode='w') as hdf5_file:

            print("FER2013 dataset creation in progress...")
            start = datetime.datetime.now().replace(microsecond=0)

            # Creates the FER2013 training, validation and testing sub-datasets
            # [TRAINING]
            hdf5_file.create_dataset("training_img", (len(self._train_X), self._image_size, self._image_size, 1), dtype='uint8', chunks=True)
            hdf5_file.create_dataset("training_label", (len(self._train_Y), 1), dtype='uint8', chunks=True)

            # [VALIATION]
            hdf5_file.create_dataset("validation_img", (len(self._valid_X), self._image_size, self._image_size, 1), dtype='uint8', chunks=True)
            hdf5_file.create_dataset("validation_label", (len(self._valid_Y), 1), dtype='uint8', chunks=True)

            # [TESTING]
            hdf5_file.create_dataset("testing_img", (len(self._test_X), self._image_size, self._image_size, 1), dtype='uint8', chunks=True)
            hdf5_file.create_dataset("testing_label", (len(self._test_Y), 1), dtype='uint8', chunks=True)

            # Creates each sub-dataset
            self.dataset_creation(hdf5_file, 'training')
            self.dataset_creation(hdf5_file, 'validation')
            self.dataset_creation(hdf5_file, 'testing')
            end = datetime.datetime.now().replace(microsecond=0)

            print("FER2013 dataset creation completed.")
            print("Duration: " + str((end - start)) + "\n")

            # Displays the new labels
            print("[NEW LABELS]\t\t0:Angry/Disgust, 1:Fear, 2:Happy, 3:Sad, 4:Surprise, 5:Neutral")
            return


    def dataset_creation(self, hdf5_file=None, operation='training'):
        """
        This method creates the dataset of the related operation e.g. training
        INPUT:
        - hdf5_file: The hdf5 file to store the data
        - operation: The dataset creation operation to execute (training, validation or testing)
        RETURN:
        - max_examples: The number of the examples created
        """
        assert hdf5_file is not None

        print("Creating the "+operation+" set...")
        index = 0
        img_tensor = None
        max_examples = 0

        # Checks which data from the FER2013 to use
        op_X = -1
        op_Y = -1
        if operation is 'training':
            op_X = self._train_X
            op_Y = self._train_Y
        elif operation is 'validation':
            op_X = self._valid_X
            op_Y = self._valid_Y
        elif operation is 'testing':
            op_X = self._test_X
            op_Y = self._test_Y
        else:
            print("'INVALID OPERATION: Please provide which dataset operation to execute i.e. training, validation or testing")
            return -1

        for i, (img, lab) in enumerate(zip(op_X, op_Y)):

            # Merges the emotions anger and disgust into one class, reduces the emotion classes by one to keep
            # consistent with 0=Angry/Disgust, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
            if (lab == 0 or lab == 1):
                lab = 1
            elif (lab > 0 and lab <= 6):
                lab -= 1

            # Stores the data (in tensor format) into the hdf5 dataset
            img_tensor = self.image_to_tensor(img)
            hdf5_file[operation+'_img'][index, :, :, :] = img_tensor
            hdf5_file[operation+'_label'][index, :] = lab
            index += 1
            max_examples += 1
            del (img); del (lab)  # Deletes the temporary image and label lists to free the memory

        return max_examples


    def image_to_tensor(self, image):
        """
        This method transforms the image to a tensor
        INPUT:
        - image: The image to transform
        RETURN:
        - tens_image: The image tensor
        """
        tens_image = cv2.resize(image, dsize=(self._image_size, self._image_size))
        tens_image = tens_image.reshape(self._image_size, self._image_size, 1)
        return tens_image


if __name__ == '__main__':

    FER2013DatasetCreation('./datasets/fer2013.csv')




