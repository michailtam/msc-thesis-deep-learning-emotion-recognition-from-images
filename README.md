# Facial Expression Recognition using Deep Learning #

The ability to predict emotions based on static or dynamic images has improved the computer vision and robotics field and remains a major research topic. Computer vision is the task of detecting or recognizing objects and persons on images or videos. 



Another important task is to predict the emotion of a person’s face, which is called FER (facial emotion recognition). For example, in health care, a device or robot can observe the state of a person and call the ambulance if the person looks to be sick. An application could play specific types of songs i.e. Soul, Blues, Rock, etc. based on the mood of a person or a suspicious person or weapon could be detected by cameras. These are only some application fields, but there are existing many more. 

## Convolutional Neural Networks
The components which are able to extract and classify visual information’s are the convolutional neural networks (CNN's). They comprise filters (kernels) and have a shallow or deep layer structure, whereas the lower layers are able to detect and extract simple visual features like lines and edges and the deeper layers more complex features like eyes, mouth, etc. If you want to learn more about these networks you can take a look at this paper: [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 

## Project description ##
To compare the performance of an older against a new CNN, two deep neural networks (DNN’s) will be used. Both networks will be trained using a method called transfer learning in which the pre-trained networks will be adapted and fine-tuned to predict the seven emotion classes (angry, disgust, fear, happy, sad, surprise and neutral). The dataset that will be used to train and evaluate the networks is the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) which is provided from the Kaggle competition and are especially suited for facial emotion recognition tasks (FER). To provide enough data while training the networks, the training set will be augmented with additional images during the execution.

## How to run the program
To run the program you have to do the following:
1. Create two folders in the directory (dataset, hdf5)
2. In the dataset folder you have to put the FER2013 dataset csv file which you have previously downloaded from Kaggle. In addition, you have to create a folder name it photo and put all your unseen photos for the test process inside it.
3. At the beginning you have to run the FER2013DatasetCreation.py file, to transform the csv content to the training, validation and test set images for the networks (this takes some time dependent on the system).
4. In the MainProc.py file you can choose by uncommenting the appropriate line, what process you want to execute (training or testing). IMPORTANT: Don't forget to leave comment the lines you don't want to execute.
5. If you want to execute the training process from the beginning you have to delete manually the hdf5 file of the network e.g. logs -> vgg16 -> epoch-15-val_acc_0.71.hdf5. 