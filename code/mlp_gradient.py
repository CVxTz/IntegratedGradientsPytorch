import json
from glob import glob
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]

        tokens = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(y, dtype=torch.float)

        return tokens, label


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 16)
        self.fy = torch.nn.Linear(16, 16)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_hat = self.fy(x)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def build_dataset(size):
    df = pd.DataFrame(
        {
            "POS1": np.random.uniform(-1, 1, size=size) * 1,
            "POS2": np.random.uniform(-1, 1, size=size) * 2,
            "POS3": np.random.uniform(-1, 1, size=size) * 3,
            "POS4": np.random.uniform(-1, 1, size=size) * 4,
            "POS5": np.random.uniform(-1, 1, size=size) * 5,
            "NEG1": np.random.uniform(-1, 1, size=size) * -1,
            "NEG2": np.random.uniform(-1, 1, size=size) * -2,
            "NEG3": np.random.uniform(-1, 1, size=size) * -3,
            "NEG4": np.random.uniform(-1, 1, size=size) * -4,
            "NEG5": np.random.uniform(-1, 1, size=size) * -5,
            "NOISE1": np.random.uniform(-1, 1, size=size) * 1,
            "NOISE2": np.random.uniform(-1, 1, size=size) * 2,
            "NOISE3": np.random.uniform(-1, 1, size=size) * 3,
            "NOISE4": np.random.uniform(-1, 1, size=size) * 4,
            "NOISE5": np.random.uniform(-1, 1, size=size) * 5,
        }
    )

    df["target"] = df.apply(lambda x: x["POS"])


if __name__ == "__main__":

    train_data = Dataset(train)
    test_data = Dataset(test)
    val_data = Dataset(val)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    model = AudioClassifier(reconstruction_weight=reconstruction_weight)

    logger = TensorBoardLogger(
        save_dir="../",
        version="Lambda=%s" % reconstruction_weight,
        name="lightning_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
        filepath="../models/",
        prefix="model_%s" % reconstruction_weight,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[DecayLearningRate()],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)
