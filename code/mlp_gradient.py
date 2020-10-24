import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        self.fy = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_hat = self.fy(x)

        return y_hat.squeeze()

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
        return torch.optim.Adam(self.parameters(), lr=1e-2)


def build_dataset(size):
    pos_size = 32
    neg_size = 32
    noise_size = 32
    pos_cols = ["POS_%s" % i for i in range(pos_size)]
    neg_cols = ["NEG_%s" % i for i in range(neg_size)]
    noise_cols = ["NOISE_%s" % i for i in range(noise_size)]

    pos = {i: np.random.uniform(-1, 1, size=size) for i in pos_cols}
    neg = {i: np.random.uniform(-1, 1, size=size) for i in neg_cols}
    noise = {i: np.random.uniform(-1, 1, size=size) for i in noise_cols}

    df = pd.DataFrame({**pos, **neg, **noise})

    df["target"] = df.apply(
        lambda x: sum(
            [x[k] * (i + 1) / pos_size for i, k in enumerate(pos_cols)]
            + [-x[k] * (i + 1) / neg_size for i, k in enumerate(neg_cols)]
        ),
        axis=1,
    )

    coefs = (
        [(i + 1) / pos_size for i, k in enumerate(pos_cols)]
        + [-(i + 1) / neg_size for i, k in enumerate(neg_cols)]
        + [0 for i, k in enumerate(noise_cols)]
    )

    return np.array(df[pos_cols + neg_cols + noise_cols]), np.array(df["target"]), coefs


def compute_integrated_gradient(batch_x, batch_blank, model):
    mean_grad = 0
    n = 100

    for i in tqdm(range(1, n + 1)):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients, mean_grad


def plot_importance(arrays, titles, output_path):
    fig, axs = plt.subplots(1, len(arrays))

    fig.set_figheight(7)
    fig.set_figwidth(10)

    for i, ((title1, title2), (array1, array2)) in enumerate(zip(titles, arrays)):
        axs[i].scatter(range(len(array1)), array1, label=title1, s=6, marker="+")
        axs[i].scatter(range(len(array2)), array2, label=title2, s=6, marker="v")
        axs[i].legend()
        axs[i].set_xlabel('Feature #')

    fig.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    batch_size = 128
    epochs = 32

    X_train, Y_train, coefs = build_dataset(20000)
    X_val, Y_val, _ = build_dataset(2000)
    X_test, Y_test, _ = build_dataset(2000)

    print(Y_train)

    train_data = Dataset(X_train, Y_train)
    test_data = Dataset(X_val, Y_val)
    val_data = Dataset(X_test, Y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    model = MLPClassifier(input_size=X_train.shape[1])

    logger = TensorBoardLogger(
        save_dir="../",
        name="lightning_logs",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)

    torch.save(model.state_dict(), "../output/model.pth")

    model.load_state_dict(torch.load("../output/model.pth"))

    i = 1
    x = X_test[i : (i + 1), :]
    x_b = np.zeros((1, X_test.shape[1]))
    batch_x = torch.from_numpy(x).float()
    batch_blank = torch.from_numpy(x_b).float()

    if torch.cuda.is_available():
        model.to("cuda")
        batch_x = batch_x.to("cuda")
        batch_blank = batch_blank.to("cuda")

    integrated_gradient, mean_grad = compute_integrated_gradient(
        batch_x, batch_blank, model
    )

    integrated_gradient = integrated_gradient.squeeze().cpu().data.numpy()
    mean_grad = mean_grad.squeeze().cpu().data.numpy()

    true_influence = np.array(coefs) * (x.squeeze() - x_b.squeeze())

    arrays = [
        [coefs, mean_grad.tolist()],
        [true_influence, integrated_gradient.tolist()],
    ]
    titles = [
        ["True Attribution / (x - x')", "Integrated Gradients / (x - x')"],
        ["True Attribution", "Integrated Gradients"],
    ]
    plot_importance(arrays, titles, "../output/mlp_integrated_gradient.png")
