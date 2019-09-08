#
#       Author:                 Michail Tamvakeras
#       Postgraduate course:    MSc in Intelligent Information Systems (MIIS)
#       University:             University of the Aegean
#       Department:             Information and Communication Systems Engineering
#
#       MSc Thesis:             Facial emotion recognition using deep learning
#

from FERManager import *
from FerVGG16 import *
from FerInceptionV3 import *


if __name__ == '__main__':

    # Creates the pre-trained Inception-v3 network
    inception_v3 = FerInceptionV3(
        64,             # Mini-batch size
        1E-4,           # Initial learning rate
        100,            # Max epochs
        'inception-v3', # The model name
        100,            # Image input size (orig: 299x299x3) IMPORTANT: For testing image size must be the original
        'rgb')

    # Creates the pre-trained VGG16 network
    vgg16 = FerVGG16(
        64,             # Mini-batch size
        1E-4,           # Initial learning rate
        100,            # Max epochs
        'vgg16',        # The model name
        100,            # Image input size  (orig: 224x224x3) IMPORTANT: For testing image size must be the original
        'rgb')

    # Creates the FER neural network manager, who manages the networks and the whole FER process
    nn_manager = FerManager()
    nn_manager.add_network(vgg16)
    nn_manager.add_network(inception_v3)

    # Executes the training process
    # nn_manager.start_training('vgg16')
    # nn_manager.start_training('inception-v3')

    # Executes the testing process
    nn_manager.test("vgg16")
    # nn_manager.test("inception-v3")

    # Predicts the emotion of the images of the specified folder
    # nn_manager.predict("./datasets/photos/", "vgg16")
    # nn_manager.predict("./datasets/photos/", "inception-v3")











