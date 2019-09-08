#
#       Project:             Facial emotion recognition using deep learning
#       Developer:           Michail Tamvakeras
#

import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Just disables the warning, doesn't enable AVX/FMA

from FerVGG16 import *
from FerInceptionV3 import *
import tensorflow as tf
import keras.backend.tensorflow_backend as tf_backend
from VisualizationTools import plot_barchart, setup_plot


class FerManager:
    '''
    This class contains the management of the facial emotion recognition process (FER).
    '''

    _num_emotions = 6          # Emotions (1=Angry/Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    _log_dir = './logs/'


    def __init__(self):
        self._nn_dict = {}     # Creates an empty network dictionary
        return


    def enable_gpu_support(self):
        # Checks for GPU support on the system
        self._gpus = tf_backend._get_available_gpus()
        if len(self._gpus) is not 0:
            tf.Session(config=tf.ConfigProto(log_device_placement=True)) # logs the GPU
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            tf_backend.set_session(sess)
        else:
            print("No GPU found on system, execution will be done using the CPU only!!!")
        return


    def add_network(self, network=None):
        """
        This method adds a new network to the network list
        network: The network tha has to be added to the network list
        INPUT:
        - network: The network
        """

        # Checks if the given net is subclass of CNNetwork
        assert issubclass(type(network), CNNetwork)
        self._nn_dict[network.get_name()] = network
        return


    def get_network_by_name(self, model_name=None):
        """
        This method loads the stored network by its name from the neural network list
        model_name: The name of the neural network to load
        INPUT:
        - model_name: The name of the network to load
        RETURN:
        - The specified network in the network list
        """
        assert model_name is not None
        return self._nn_dict[model_name]


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


    def start_training(self, network_type):
        """
        This method executes the training and validation process of the models
        INPUT:
        - network_type: The type of the network to train
        """

        self.enable_gpu_support()   # Enables the GPU support when it's available

        if network_type not in self._nn_dict:
            raise ValueError("The given network could not be found in the neural network list!!!.")
        network = self.get_network_by_name(network_type)

        # Checks if the log folder for the VGG16 network exists
        if network.get_name() == 'vgg16':
            if os.path.exists(self._log_dir + "vgg16") == False:
                os.makedirs(self._log_dir + "/vgg16/train/")

        # Checks if the log folder for the Inception-v3 network exists
        if network.get_name() == 'inception-v3':
            if os.path.exists(self._log_dir + "inception_v3") == False:
                os.makedirs(self._log_dir + "/inception_v3/train/")

        print("Training of the "+network_type+" network starts...")
        network.build(self._num_emotions)
        network.training(augmentation=True, early_stopping=True, decay_rate=0.5)  # Decays 50% the learning rate
        return


    def predict(self, dataset_path, network=None):
        """
        This method classifies an unknown image
        INPUT:
        - dataset_path: The path of the dataset
        - network: The network to predict
        """
        assert network is not None

        image_list = os.listdir(dataset_path)
        print("Prediction using the", network,"model...")
        vgg16 = None
        test_model_dir = ""
        inception_resnet_v2 = None

        # Checks for the VGG16 network
        if network == 'vgg16':
            vgg16 = self.get_network_by_name(network)
            test_model_dir = "./logs/vgg16/train/"
        # Checks for the Inception-v3 network
        elif network == 'inception-v3':
            inception_resnet_v2 = self.get_network_by_name(network)
            test_model_dir = "./logs/inception_v3/train/"
        else:
            print("No specified network for testing found!!!")
            return

        # Checks if the test directory contains a test checkpoint
        if os.listdir(test_model_dir):
            file = [file for file in os.listdir(test_model_dir)]
            test_model = load_model(test_model_dir + str(file[0]))
        else:
            print("No saved models available, prediction cannot proceed!!!")
            return
        emotion_predictions = []

        # Predicts the unknown images
        for img_file in image_list:
            # Reads every image from the folder, converts it to grayscale
            img_orig = cv2.imread(dataset_path + img_file)
            img_gray = cv2.imread(dataset_path + img_file, 0)

            predictions = []; calc_predictions = []; y_label = []
            if vgg16:
                predictions = vgg16.predict(img_gray, test_model)
            elif inception_resnet_v2:
                predictions = inception_resnet_v2.predict(img_gray, test_model)
            # print(predictions)

            print("Image file: ", img_file, ":")
            for code, percent in enumerate(predictions):
                emotion = self.code_to_emotion(code)    # Converts the code to the emotion name

                # Checks for which emotion/percentage to print
                if code == 0: print(emotion + ":\t%1.2f%%" % (np.float(percent)))
                elif code == 1: print(emotion + ":\t\t\t%1.2f%%" % (np.float(percent)))
                elif code == 2: print(emotion + ":\t\t\t%1.2f%%" % (np.float(percent)))
                elif code == 3: print(emotion + ":\t\t\t%1.2f%%" % (np.float(percent)))
                elif code == 4: print(emotion + ":\t\t%1.2f%%" % (np.float(percent)))
                elif code == 5: print(emotion + ":\t\t%1.2f%%" % (np.float(percent)))
                else: print("Invalid Code!!!")
                calc_predictions.append(percent)

            highest = self.code_to_emotion(np.argmax(calc_predictions))
            print("Highest:", highest, "\n")
            emotion_predictions.append(highest)

            cv2.imshow(highest, img_orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.plot_statistics(emotion_predictions)  # Plots the statistics of the prediction
        return


    # TODO: Implementation of the testing process
    def test(self, network=None):
        """
        This method uses the test set to evaluate the given network
        INPUT:
        - dataset_path: The path of the dataset
        - network: The network to predict
        """
        assert network is not None

        self.enable_gpu_support()  # Enables the GPU support if it's available

        print("Testing using the", network, "model...")
        vgg16 = None
        test_model_dir = ""
        inception_resnet_v2 = None

        # Checks for the VGG16 network
        if network == 'vgg16':
            vgg16 = self.get_network_by_name(network)
            test_model_dir = "./logs/vgg16/train/"
        # Checks for the Inception-v3 network
        elif network == 'inception-v3':
            inception_resnet_v2 = self.get_network_by_name(network)
            test_model_dir = "./logs/inception_v3/train/"
        else:
            print("No specified network for testing found!!!")
            return

        # Checks if the test directory contains a test checkpoint
        if os.listdir(test_model_dir):
            file = [file for file in os.listdir(test_model_dir)]
            test_model = load_model(test_model_dir + str(file[0]))
        else:
            print("No saved models available, prediction cannot proceed!!!")
            return

        result = []
        if vgg16:
            result = vgg16.testing(self._num_emotions, test_model)
        elif inception_resnet_v2:
            result = inception_resnet_v2.testing(self._num_emotions, test_model)
        print("Test Accuracy: {:.2f}%".format(result))
        return


    def plot_statistics(self, results):
        """
        This method prints the statistics of the prediction
        INPUT:
        - results: The results of the prediction to plot
        """

        # Prints a bar chart
        emotion_names = ["Angry,\nDisgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        occurence = []
        for emotion in emotion_names:
            number =   results.count(emotion)
            occurence.append(number)
        setup_plot(font_size=22, show_grid=False)
        plot_barchart(emotion_names, occurence)
        return