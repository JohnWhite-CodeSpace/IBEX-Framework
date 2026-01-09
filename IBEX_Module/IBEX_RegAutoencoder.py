import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
FEATURE_IDXS = [
    0,      # sum
    4,      # rate
    5,      # cos(RA)
    6,      # sin(RA)
    7,      # R_RA
    # 8,    # cos(phase)
    # 9,    # sin(phase)
    10,   # R(phase)
    11,   # mean(X)
    12,   # mean(Y)
    13,   # mean(Z)
    14,   # std(X)
    15,   # std(Y)
    16,   # std(Z)
    17,   # mean(R)
    18    # std(R)
]
class IBEX_RegAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dec_output_dim: int =208, reg_output_dim: int =16, latent_dim: int=32):
        super(IBEX_RegAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.02),

            nn.Linear(64, latent_dim),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, reg_output_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.02),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02),

            nn.Linear(128, dec_output_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        prediction = self.regression_head(latent)
        return reconstruction, prediction, latent

class IBEXDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_aggregated_data(data_dir: str, target_channel: int, channels:int = 14, feature_idx: list = FEATURE_IDXS):
    try:
        channel_data = {}
        lengths = []
        for i in range(1, channels+1):
            path = os.path.join(data_dir, f"channel_{i}_good_data_aggregated.txt")
            data = np.loadtxt(path, skiprows=1)
            channel_data[i] = data
            lengths.append(len(data))
        min_len = min(lengths)
        X_list = []
        y_list = []
        for i in range(min_len):
            x_row = []
            y_row = None
            for ch in range(1, channels+1):
                features = channel_data[ch][i, feature_idx]
                if ch == target_channel:
                    y_row = features
                x_row.extend(features)
            X_list.append(x_row)
            y_list.append(y_row)
        return X_list, y_list
    except Exception as ex:
        print(ex)
        return None, None

def build_dataloaders(X_list, y_list, batch_size: int = 256, test_size: float = 0.2, val_size: float = 0.1,random_state: int = 42,):
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    if val_size > 0.0:
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=random_state, shuffle=True)
    else:
        X_val, y_val = None, None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test)

    if X_val is not None:
        X_val = scaler_X.transform(X_val)
        y_val = scaler_y.transform(y_val)

    train_dataset = IBEXDataset(X_train, y_train)
    test_dataset  = IBEXDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if X_val is not None:
        val_dataset = IBEXDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    return train_loader, val_loader, test_loader, scaler_X, scaler_y

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    std  = np.std(y_true)
    rmse_std = rmse / std if std > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "RMSE/STD": rmse_std}

def run_epoch(model, loader, optimizer=None, lambda_reg=1.0, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_rec  = 0.0
    total_reg  = 0.0

    y_true_all = []
    y_pred_all = []

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            x_hat, y_hat, _ = model(X)

            loss_rec= F.mse_loss(x_hat, X)
            loss_reg = F.mse_loss(y_hat, y)
            loss = loss_rec + lambda_reg * loss_reg

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_rec+= loss_rec.item() * X.size(0)
            total_reg += loss_reg.item() * X.size(0)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(y_hat.detach().cpu().numpy())

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    return {"loss": total_loss / len(loader.dataset),"loss_rec": total_rec / len(loader.dataset),
        "loss_reg": total_reg / len(loader.dataset),"y_true": y_true_all,"y_pred": y_pred_all}

def train_reg_autoencoder(model, train_loader, val_loader, test_loader,
    scaler_y, optimizer, epochs: int = 200, lambda_reg: float = 0.1, device: str = "cpu"):
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_rec": [],
        "val_rec": [],"train_reg": [], "val_reg": []}

    best_val_loss = np.inf
    best_state = None
    for epoch in range(1, epochs + 1):
        train_out = run_epoch(model, train_loader, optimizer=optimizer, lambda_reg=lambda_reg, device=device)
        val_out = run_epoch(model, val_loader, optimizer=None, lambda_reg=lambda_reg, device=device)

        history["train_loss"].append(train_out["loss"])
        history["val_loss"].append(val_out["loss"])
        history["train_rec"].append(train_out["loss_rec"])
        history["val_rec"].append(val_out["loss_rec"])
        history["train_reg"].append(train_out["loss_reg"])
        history["val_reg"].append(val_out["loss_reg"])
        if val_out["loss"] < best_val_loss:
            best_val_loss = val_out["loss"]
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:4d}/{epochs}] "
                f"Train L={train_out['loss']:.4e} "
                f"(rec={train_out['loss_rec']:.4e}, reg={train_out['loss_reg']:.4e}) | "
                f"Val L={val_out['loss']:.4e}"
            )

    model.load_state_dict(best_state)
    train_eval = run_epoch(model, train_loader, device=device)
    val_eval   = run_epoch(model, val_loader, device=device)
    test_eval  = run_epoch(model, test_loader, device=device)

    def eval_block(name, out):
        m_norm = compute_metrics(out["y_true"], out["y_pred"])
        y_true_phys = scaler_y.inverse_transform(out["y_true"])
        y_pred_phys = scaler_y.inverse_transform(out["y_pred"])
        m_phys = compute_metrics(y_true_phys, y_pred_phys)

        return {"normalized": m_norm, "physical": m_phys}
    metrics = {"train": eval_block("train", train_eval), "val":   eval_block("val", val_eval), "test":  eval_block("test", test_eval)}
    return model, history, metrics

def get_predictions(model, loader, device="cpu"):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            _, y_hat, _ = model(X)
            y_true.append(y.cpu().numpy())
            y_pred.append(y_hat.cpu().numpy())

    return np.vstack(y_true), np.vstack(y_pred)

def plot_pred_vs_true_grid_4x4(y_true_phys, y_pred_phys, feature_names, title,):
    n_features = y_true_phys.shape[1]
    n_rows = int(np.ceil(n_features / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n_features]):
        ax.scatter(y_true_phys[:, i], y_pred_phys[:, i], s=8, alpha=0.5)

        lims = [
            min(y_true_phys[:, i].min(), y_pred_phys[:, i].min()),
            max(y_true_phys[:, i].max(), y_pred_phys[:, i].max())
        ]

        ax.plot(lims, lims, "--", color="orange", label="ideal: y=x")
        ax.set_title(feature_names[i])
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.grid(True)
        ax.legend()

    for ax in axes[n_features:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train total loss")
    plt.plot(epochs, history["val_loss"], label="Val total loss")
    plt.plot(epochs, history["train_rec"], "--", label="Train reconstruction")
    plt.plot(epochs, history["val_rec"], "--", label="Val reconstruction")
    plt.plot(epochs, history["train_reg"], ":", label="Train regression")
    plt.plot(epochs, history["val_reg"], ":", label="Val regression")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("AE + regression head – training curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

