import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

__author__ = "Jan Biały"

# Column indices in raw IBEX event tensors
COL_MET = 0 # Mission Elapsed Time
COL_RA = 1  # Right Ascension
COL_DECL = 2    # Declination
COL_PHASE = 7   # Instrument phase
COL_X = 8   # Spacecraft X position (RE)
COL_Y = 9   # Spacecraft Y position (RE)
COL_Z = 10  # Spacecraft Z position (RE)
def load_event_data(base_dir: str | Path)-> dict[int, torch.Tensor]:
    """
    Load raw IBEX event-level data for all energy channels.

    The function loads preprocessed PyTorch tensors corresponding to
    individual IBEX-Hi (channels 1–6) and IBEX-Lo (channels 7–14) energy channels.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Base directory containing ``Hi_data`` and ``Lo_data`` subdirectories.

    Returns
    -------
    dict[int, torch.Tensor]
        Dictionary mapping global channel indices (1–14) to event tensors
        of shape ``(N_events, K_features)``.

    Raises
    ------
    FileNotFoundError
        If any required channel file is missing.
    """
    base_dir = Path(base_dir)
    data = {}

    # HI: channels 1–6
    for ch in range(1, 7):
        path = base_dir / "Hi_data" / f"hi_hex_channel_{ch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        data[ch] = torch.load(path)

    # LO: channels 7–14
    for ch in range(1, 9):
        path = base_dir / "Lo_data" / f"lo_hex_channel_{ch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        data[ch + 6] = torch.load(path)

    return data

def aggregate_events_to_bins(event_data: dict[int, torch.Tensor], time_bin: float = 600.0)-> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate raw ENA events into fixed-duration time bins.

    For each time interval, the function computes:

    * ENA detection rates for all 14 energy channels,
    * auxiliary physical and geometrical conditions derived from
      a reference channel.

    Time bins with missing data in any channel are discarded.

    Parameters
    ----------
    event_data : dict[int, torch.Tensor]
        Dictionary mapping channel indices to event-level tensors.
    time_bin : float, optional
        Width of the time bin in seconds. Default is 600 s.

    Returns
    -------
    rates : np.ndarray
        Array of shape ``(N_bins, 14)`` containing ENA detection rates
        for all energy channels.
    conditions : np.ndarray
        Array of shape ``(N_bins, N_conditions)`` containing aggregated
        physical and geometrical features per time bin.
    """
    #GLOBAL TIME RANGE
    all_times = torch.cat(
        [data[:, COL_MET] for data in event_data.values()]
    ).cpu().numpy()

    t_min, t_max = all_times.min(), all_times.max()
    bins = np.arange(t_min, t_max, time_bin)

    rates = []
    conditions = []

    ref_ch = list(event_data.keys())[0]

    #ITERATION THROUGH TIME BINS
    for t0 in bins:
        t1 = t0 + time_bin

        rate_vec = np.zeros(14, dtype=np.float32)
        valid = True

        #RATE PER CHANNEL
        for ch in range(1, 15):
            data = event_data[ch]
            met = data[:, COL_MET]

            mask = (met >= t0) & (met < t1)
            n = int(mask.sum().item())

            if n == 0:
                valid = False
                break

            rate_vec[ch - 1] = n / time_bin

        if not valid:
            continue

        #CONDITIONS (from reference channel)
        ref = event_data[ref_ch]
        met_ref = ref[:, COL_MET]
        mref = (met_ref >= t0) & (met_ref < t1)

        if mref.sum() == 0:
            continue

        ra = np.deg2rad(ref[mref, COL_RA].cpu().numpy())
        decl = np.deg2rad(ref[mref, COL_DECL].cpu().numpy())
        ph = ref[mref, COL_PHASE].cpu().numpy()
        x = ref[mref, COL_X].cpu().numpy()
        y = ref[mref, COL_Y].cpu().numpy()
        z = ref[mref, COL_Z].cpu().numpy()
        dir_x = np.cos(decl) * np.cos(ra)
        dir_y = np.cos(decl) * np.sin(ra)
        dir_z = np.sin(decl)
        R = np.sqrt(x**2 + y**2 + z**2)

        cond = np.array([
            # mean ENA direction
            np.mean(dir_x),
            np.mean(dir_y),
            np.mean(dir_z),
            # geometry of scanned space in search of ENA
            np.mean(np.cos(2 * np.pi * ph)),
            np.mean(np.sin(2 * np.pi * ph)),
            # mean distance from Earth
            np.mean(R),
            # standard deviation of the orbit in time bin (time range)
            np.std(R)
        ], dtype=np.float32)

        rates.append(rate_vec)
        conditions.append(cond)
    norm = np.sqrt(cond[0] ** 2 + cond[1] ** 2 + cond[2] ** 2)
    print("Mean direction norm:", norm)
    return np.stack(rates), np.stack(conditions)

def save_aggregated_dataset(output_dir: str | Path, rates: np.ndarray, conditions: np.ndarray, time_bin: float)-> None:
    """
    Save aggregated rate and condition datasets to disk.

    The function stores NumPy arrays along with a JSON metadata file
    describing the aggregation parameters.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Output directory for the aggregated dataset.
    rates : np.ndarray
        Array containing per-channel ENA detection rates.
    conditions : np.ndarray
        Array containing aggregated physical and geometrical features.
    time_bin : float
        Time bin width used during aggregation (in seconds).

    Returns
    -------
    None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "rates.npy", rates)
    np.save(output_dir / "conditions.npy", conditions)

    meta = {
        "time_bin_sec": time_bin,
        "n_samples": len(rates),
        "rate_shape": rates.shape,
        "condition_shape": conditions.shape
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def init_dataset_preparation_pipeline(base_dir: str, out_dir: str, regimes: list[str] = ["BeforePerigeeChange", "AfterPerigeeChange"]) -> None:
    """
    Execute the full dataset preparation pipeline for multiple mission regimes.

    For each regime, the function loads raw event data, aggregates it into
    time bins, and saves the resulting datasets to disk.

    Parameters
    ----------
    base_dir : str
        Base directory containing regime-specific input data.
    out_dir : str
        Output directory for aggregated datasets.
    regimes : list of str, optional
        List of regime identifiers to process.

    Returns
    -------
    None
    """
    for regime in regimes:
        base = f"{base_dir}/{regime}"
        out = f"{out_dir}/{regime}"
        events = load_event_data(base)
        rates, conds = aggregate_events_to_bins(events, time_bin=600.0)
        save_aggregated_dataset(out, rates, conds, time_bin=600.0)
    print(f'All events have been successfully aggregated and saved in {out_dir}.')

class RateAutoencoder(nn.Module):
    """
    Autoencoder model for ENA detection rate vectors.

    The model compresses 14-dimensional per-channel ENA rate vectors
    into a low-dimensional latent representation and reconstructs
    the original rates from it.

    ------
    """
    def __init__(self, input_dim: int = 14, latent_dim: int = 4):
        """
        Initialize the autoencoder architecture.

        Parameters
        ----------
        input_dim : int, optional
            Dimensionality of the input rate vector (default: 14 channels).
        latent_dim : int, optional
            Dimensionality of the latent space.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.02),

            nn.Linear(32, 16),
            nn.LeakyReLU(0.02),

            nn.Linear(16, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.02),

            nn.Linear(16, 32),
            nn.LeakyReLU(0.02),

            nn.Linear(32, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed input tensor.
        z : torch.Tensor
            Latent representation.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None = None, n_epochs: int = 200, lr: float = 1e-3, device: str = "cpu", verbose: bool = True)-> dict[str, list[float]]:
    """
    Train an autoencoder model using mean squared error loss.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model to be trained.
    train_loader : DataLoader
        DataLoader providing training samples.
    val_loader : DataLoader, optional
        DataLoader providing validation samples.
    n_epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Learning rate.
    device : str, optional
        Device identifier (``'cpu'`` or ``'cuda'``).
    verbose : bool, optional
        If True, training progress is printed.

    Returns
    -------
    dict
        Dictionary containing training and validation loss histories.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, n_epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0

        for x in train_loader:
            x = x.to(device)

            x_hat, _ = model(x)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # VALIDATION
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    x_hat, _ = model(x)
                    loss = criterion(x_hat, x)
                    val_loss += loss.item() * x.size(0)

            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

        # LOG
        if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == n_epochs):
            if val_loader is not None:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE: {train_loss:.6e} | "
                    f"Val MSE: {val_loss:.6e}"
                )
            else:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE: {train_loss:.6e}"
                )

    return history

def load_autoencoder_data(path: str | Path)-> tuple[np.ndarray, StandardScaler]:
    """
    Load and preprocess aggregated rate data for autoencoder training.

    The preprocessing includes logarithmic transformation and per-channel
    standardization.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the directory containing ``rates.npy``.

    Returns
    -------
    X : np.ndarray
        Preprocessed rate matrix.
    scaler : StandardScaler
        Fitted scaler used for standardization.
    """
    path = Path(path)

    rates = np.load(path / "rates.npy")

    # log-transform
    X = np.log10(rates + 1e-6)

    # standardization (per channel)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    return Xs, scaler

def make_dataloaders(X: np.ndarray, batch_size: int = 256, val_frac: float = 0.2) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from a NumPy array.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    batch_size : int, optional
        Batch size.
    val_frac : float, optional
        Fraction of data reserved for validation.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for training.
    val_loader : DataLoader
        DataLoader for validation.
    """
    tensor = torch.tensor(X, dtype=torch.float32)

    n_val = int(len(tensor) * val_frac)
    n_train = len(tensor) - n_val

    train_ds, val_ds = random_split(tensor, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def plot_training_history(history: dict[str, list[float]], title: str)-> None:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    history : dict
        Dictionary containing loss histories.
    title : str
        Plot title.

    Returns
    -------
    None
    """
    plt.figure(figsize=(6, 4))

    plt.plot(history["train_loss"], label="Train")
    if len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label="Validation")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def reconstruction_mse(model: nn.Module, X: np.ndarray, device: str = "cpu")-> float:
    """
    Compute global reconstruction mean squared error.

    Parameters
    ----------
    model : torch.nn.Module
        Trained autoencoder model.
    X : np.ndarray
        Input data matrix.
    device : str, optional
        Computation device.

    Returns
    -------
    float
        Global reconstruction MSE.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        X_hat, _ = model(X_t)
        mse = torch.mean((X_hat - X_t) ** 2).item()
    return mse

def per_channel_mse(model: nn.Module, X: np.ndarray, device: str = "cpu")-> np.ndarray:
    """
    Compute per-channel reconstruction mean squared error.

    Parameters
    ----------
    model : torch.nn.Module
        Trained autoencoder model.
    X : np.ndarray
        Input data matrix.
    device : str, optional
        Computation device.

    Returns
    -------
    np.ndarray
        Array of per-channel MSE values.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        X_hat, _ = model(X_t)
        mse = torch.mean((X_hat - X_t) ** 2, dim=0).cpu().numpy()
    return mse

def plot_per_channel_mse(mse: np.ndarray, title: str)-> None:
    """
    Plot per-channel reconstruction error.

    Parameters
    ----------
    mse : np.ndarray
        Per-channel MSE values.
    title : str
        Plot title.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(mse) + 1), mse)
    plt.xlabel("Channel")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()