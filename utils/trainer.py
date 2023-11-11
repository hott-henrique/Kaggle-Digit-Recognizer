import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer(object):

    def __init__(self, model: nn.Module,
                       criterion: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       metric: nn.Module,
                       device: torch.device):
        super().__init__()
        self.device = device

        self.model = model
        self.criterion = criterion
        self.optmizer = optimizer

        self.metric = metric

    def train(self, train_dataloader: DataLoader):
        self.current_dataloader = train_dataloader

        self.model.train()

        for batch, (X, y) in enumerate(train_dataloader):
            X: torch.Tensor = X.to(self.device)
            y: torch.Tensor = y.to(self.device)

            self.optmizer.zero_grad()

            y_hat: torch.Tensor = self.model(X)

            loss: torch.Tensor = self.criterion(y_hat, y)

            loss.backward()

            self.optmizer.step()

            self.on_train_batch_loss(batch=batch, loss=loss)

        self.post_train()

    def test(self, test_dataloader: DataLoader):
        self.current_dataloader = test_dataloader

        self.model.eval()

        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                X: torch.Tensor = X.to(self.device)
                y: torch.Tensor = y.to(self.device)

                y_hat: torch.Tensor = self.model(X)

                metric: torch.Tensor = self.metric(y_hat, y)

                self.on_test_batch_metric(batch=batch, metric=metric)

        self.post_test()

    def on_train_batch_loss(self, batch: int, loss: torch.Tensor):
        pass

    def on_test_batch_metric(self, batch: int, metric: torch.Tensor):
        pass

    def post_train(self):
        pass

    def post_test(self):
        pass
