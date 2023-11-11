import torch, mlflow

from torch import nn
from torch.utils.data import DataLoader


def num_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_signature(model: nn.Module, test_dataloader: DataLoader):
    with torch.no_grad():
        X: torch.Tensor = next(iter(test_dataloader))

        signature = mlflow.models.infer_signature(X.numpy(), model(X).numpy())

        return signature
