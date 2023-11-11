import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class KaggleSubmitter(object):

    def __init__(self, model: nn.Module, work_dir: str = '.', submission_file: str = 'submission.csv'):
        self.model = model

        self.work_dir = work_dir
        self.submission_file = submission_file
        self.submission_path = os.path.join(self.work_dir, self.submission_file)

    def blind_test(self, test_dataloader: DataLoader, device: torch.device):
        self.current_dataloader = test_dataloader

        self.model.eval()

        with torch.no_grad():
            for batch, X in enumerate(test_dataloader):
                X: torch.Tensor = X.to(device)

                y_hat: torch.Tensor = self.model(X)

                self.on_blind_test_batch_output(batch=batch, output=y_hat)

        self.post_blind_test()

    def on_blind_test_batch_output(self, batch: int, output: torch.Tensor):
        pass

    def post_blind_test(self):
        pass
