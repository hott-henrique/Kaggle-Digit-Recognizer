import torch
import utils


class DigitsRecognizerSubmitter(utils.submitter.KaggleSubmitter):

    def on_blind_test_batch_output(self, batch: int, output: torch.Tensor):
        with open(self.submission_path, mode='a') as f:
            if f.tell() == 0:
                print('ImageId', 'Label', sep=',', file=f)

            for id, logits in enumerate(output, start=1):
                y_hat = logits.argmax()
                print(self.current_dataloader.batch_size * batch + id, y_hat.item(), sep=',', file=f)
