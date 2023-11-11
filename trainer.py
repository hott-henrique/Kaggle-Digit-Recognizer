import torch
import torch.nn as nn

import mlflow

import utils


class ClassifierTrainer(utils.trainer.Trainer):

    def __init__(self, model: nn.Module,
                       criterion: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device):
        super().__init__(model, criterion, optimizer, utils.metrics.accuracy, device)

        self.accuracy_accumulator = 0

    def reset(self):
        self.accuracy_accumulator = 0

    def on_train_batch_loss(self, batch: int, loss: torch.Tensor):
        mlflow.log_metric('loss', loss, batch)

    def on_test_batch_metric(self, batch: int, metric: torch.Tensor):
        self.accuracy_accumulator += metric.item()

    def post_test(self):
        accuracy = (self.accuracy_accumulator / len(self.current_dataloader.dataset)) * 100.0
        mlflow.log_metric('accuracy', accuracy)
