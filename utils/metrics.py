import torch

def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    return (y_hat.argmax(1) == y).sum()
