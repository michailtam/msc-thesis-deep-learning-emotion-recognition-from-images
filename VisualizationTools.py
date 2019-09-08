#
#       Project:             Facial emotion recognition using deep learning
#       Developer:           Michail Tamvakeras
#

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def setup_plot(font_size=18, show_grid=True):
    """
    Initializes the plot
    INPUT:
    - font_size: The font size
    - show_grid: Flag to show/hide the grid
    """
    plt.rcParams.update({'font.size': font_size})   # Sets the font size
    plt.clf()
    if show_grid:
        plt.grid()  # Shows the grid if desired
    return


def plot_function(interval, step, func=None, color='blue'):
    """
    This method plots the given function
    INPUT:
    - interval: The interval of the x-axis
    - step: The step for the x-axis
    - func: The function to plot
    - color: Color to print the function
    IMPORTANT: The function to plot, has to be defined before
    """
    assert func is not None

    x = np.arange(interval[0], interval[1], step)
    plt.plot(x, func(x), color=color)
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    plt.show()
    return


def plot_learning_rate_history(learning_rates=None):
    """
    This method plots the learning rate history
    INPUT:
    - training_losses: The learning rate history to plot
    """
    x_axis = [val for val in range(len(learning_rates))]
    plt.locator_params(integer=True) # Prints the x-axis values as integers
    plt.plot(x_axis, learning_rates, '#EE9A00') # Orange
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()
    return


def plot_loss_history(training_losses=None, validation_losses=None):
    """
    This method plots the loss history of the desired operations (training, validation, testing)
    INPUT:
    - training_losses: The given training losses to plot as a list
    - validation_losses: The given validation losses to plot as a list
    """
    legend = []
    x_axis = [val for val in range(len(training_losses))]
    plt.locator_params(integer=True) # Prints the x-axis values as integers
    if training_losses is not None: # Checks for plotting the training loss
        plt.plot(x_axis, training_losses, 'b-')
        legend.append('Training')
    if validation_losses is not None: # Checks for plotting the validation loss
        plt.plot(x_axis, validation_losses, 'g-')
        legend.append('Validation')
    plt.legend(legend, loc='upper right')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return


def plot_accuracy_history(training_acc=None, validation_acc=None):
    """
    This method plots the loss history of the desired operations (training, validation, testing)
    INPUT:
    - training_acc: The given training accuracy to plot as a list
    - validation_acc: The given validation accuracy to plot as a list
    """
    legend = []
    x_axis = [val for val in range(len(training_acc))]
    plt.locator_params(integer=True)  # Prints the x-axis values as integers
    if training_acc is not None:  # Checks for plotting the training loss
        plt.plot(x_axis, training_acc, 'b-')
        legend.append('Training')
    if validation_acc is not None:  # Checks for plotting the validation loss
        plt.plot(x_axis, validation_acc, 'g-')
        legend.append('Validation')
    plt.legend(legend, loc='lower right')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    return


def plot_barchart(x_values, y_values):
    """
    This method plots a bar chart of the given distribution
    INPUT:
    -
    """
    y = np.arange(len(x_values))
    plt.bar(y, y_values, align='center', color='SkyBlue')
    plt.xticks(y, x_values)
    plt.ylabel('Number')
    plt.xlabel('Emotion')
    plt.title('Prediction')
    plt.show()
    return


def plot_confusion_matrix(target_labels, predicted_labels, class_names, plot_names, invalid_names,
    title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix, based on the sklearn source:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    #sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py.
    INPUT:
    - target_labels: The target labels
    - predicted_labels: The predicted labels
    - class_names: The names of the emotion classes
    - plot_names: Names to plot
    - invalid_names: The labels which are invalid (labels that are not predicted)
    - title: Title of the confusion matrix
    - cmap: The color distribution of the confusion matrix
    """

    # Calculates the confusion matrix including invalid labels
    invalid_values = [-1] * len(invalid_names)
    cm = confusion_matrix(target_labels+invalid_names, predicted_labels+invalid_values, labels=class_names)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(plot_names))
    plt.xticks(tick_marks, plot_names)
    plt.yticks(tick_marks, plot_names, rotation=45)

    # print(cm)
    # Prints row by row the values in the confusion matrix
    for i, name in zip(range(len(class_names)), class_names):
        for j in range(len(class_names)):
            plt.text(i,j, str(cm[i][j]))
    plt.tight_layout()
    plt.ylabel('Predicted')
    plt.xlabel('Target')
    plt.show()
    return