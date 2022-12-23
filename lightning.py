import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score
from torch.utils.data import DataLoader
from rnn import RNN
from cnn import ECG_CNN
from dataset import MITBIH_DATASET


class LitECG(pl.LightningModule):
    def __init__(self, config):
        super(LitECG, self).__init__()
        self.save_hyperparameters()

        self.config = config
        self.lr = config["training"]["lr"]
        self.batch_size = config["training"]["batch_size"]
        self.is_spectrogram = config["training"]["is_spectrogram"]
        self.spectrogram = config["spectrogram"]

        self.model_type = self.config["training"]["model"]
        self.model_config = self.config[f"{self.model_type}_model"]
        num_classes = self.model_config["num_classes"]

        if self.model_type == "rnn":
            self.model = RNN(
                **self.model_config
                )
        else:
            self.model = ECG_CNN(
                **self.model_config
                )  

        self.train_dataset = MITBIH_DATASET(config["data"]["train_path"], split=config["data"]["train_val_split"])
        self.val_dataset = MITBIH_DATASET(config["data"]["train_path"], val=True, split=config["data"]["train_val_split"])
        self.test_dataset = MITBIH_DATASET(config["data"]["test_path"], test=True)

        metrics = MetricCollection([Accuracy(task="multiclass", num_classes=num_classes), Precision(task="multiclass", num_classes=num_classes), Recall(task="multiclass", num_classes=num_classes), F1Score(task="multiclass", num_classes=num_classes)])
        self.metrics = {"train": metrics.clone("train"), "valid": metrics.clone("valid"), "test": metrics.clone("test")}
        
    def forward(self, x):
        if self.is_spectrogram:
            x = x.squeeze(-1)
            x = torch.stft(x, self.spectrogram["window_size"], onesided=True, window=torch.hann_window(self.spectrogram["window_size"]), return_complex=True)
            x = x.permute(0, 2, 1)
            x = x.abs()

        if self.model_type == "cnn":
            x = x.permute(0, 2, 1)
        return self.model(x)

    def step(self, batch, type):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log_dict({f"{type}_loss": loss, **self.metrics[type](y_hat, y)})

        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def validation_step(self, batch, _):
        return self.step(batch, "valid")

    def test_step(self, batch, _):
        return self.step(batch, "test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def infer(self, x):
        self.model.eval()

        y = self(x)
        clas = torch.argmax(y, dim=-1).long()
        return clas.numpy()
