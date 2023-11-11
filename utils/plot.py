import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

from matplotlib.axes import Axes

from sklearn.metrics import confusion_matrix as calc_confusion_matrix


def image(ax: Axes, img: torch.Tensor) -> None:
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False, top=False,
                   left=False, right=False,
                   labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False)

    ax.imshow(img)

def latent_space(ax: Axes, vector: torch.Tensor, normalize: bool = False) -> None:
    with torch.no_grad():
        ax.tick_params(axis='x',
                       which='both',
                       bottom=False, top=False,
                       labelbottom=False)

        if normalize:
            vector = nn.functional.normalize(vector.flatten(), p=2.0, dim=0)

        ax.bar(range(len(vector)), vector)

        if normalize:
            ax.set_yticks(np.linspace(-1, 1, num=5))

def confusion_matrix(y: np.ndarray, y_hat: np.ndarray, classes: list[str]):
    matrix = calc_confusion_matrix(y, y_hat)

    df = pd.DataFrame(matrix / np.sum(matrix, axis=1),
                      index=classes,
                      columns=classes)

    fig, axes = plt.subplots()

    fig.suptitle("Confusion Matrix")

    im = axes.imshow(df, cmap='hot', interpolation='nearest')

    fig.colorbar(im)

    axes.set_xticks(range(len(classes)), classes, rotation=90)
    axes.set_yticks(range(len(classes)), classes)

def classification_scores(ax: Axes, classes, scores):
    ticks = range(len(classes))

    ax.barh(ticks, scores, align='center')
    ax.set_yticks(ticks, labels=classes, rotation=65)

    ax.set_xticks(np.linspace(0, 1, 5))

