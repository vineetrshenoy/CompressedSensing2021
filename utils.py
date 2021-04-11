import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# MNIST class names
classes = tuple([str(x) for x in range(10)])


def plot_results(model):
    if not (os.path.isdir("outputs")):
        os.mkdir("outputs")
    # Plot validation accuracy and loss on a single plot
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax3) = plt.subplots(2, 1)
    x_val = np.arange(len(model.validation_accuracies))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val. Loss', color=color)
    ax1.plot(x_val, np.array(model.validation_losses), color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val. Accuracy', color=color)
    ax2.plot(x_val, model.validation_accuracies, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)

    # Plot training losses in another subplot
    x = np.linspace(0, len(model.validation_accuracies) - 1, len(model.training_losses))
    color = 'tab:red'
    ax3.plot(x, model.training_losses, color=color)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Loss')
    ax3.grid(True)

    fig.tight_layout()
    plt.savefig("outputs/training_plot.png", dpi=500)
    plt.show()

    # Plot confusion matrix
    plt.figure()
    plt.rcParams.update({'font.size': 5})
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=model.cm, display_labels=classes)
    cm_plot.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Total Accuracy = %.3f%%)' % (100 * model.overall_accuracy))
    plt.savefig("outputs/confusion_matrix.png", dpi=500)