import seaborn as sns
import pandas as pd
import  torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.io.sas.sas_constants import os_version_number_offset
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from sklearn.decomposition import PCA

#################################################################
#################### DATA LOADERS AND HANDLING ##################
#################################################################
def load_data(path, max_rows=None):
    data = np.loadtxt(path, max_rows=max_rows, skiprows=1)
    counts = data[:, 0]
    start_times = data[:, 2]
    end_times = data[:, 3]
    mid_times = start_times + (start_times + end_times) / 2
    return counts, mid_times

def load_dataset(base_dir):
    base_dir = Path(base_dir)
    rates = np.load(base_dir / "rates.npy")
    conds = np.load(base_dir / "conditions.npy")
    return rates, conds

def load_channel_rate(path):
    data = np.loadtxt(path, skiprows=1)
    counts = data[:, 0]
    dt = data[:, 1]
    rate = counts / dt
    return rate

def load_channel_scalar(path):
    return torch.load(path)[:, 1].numpy()

def load_datasets(DATA_DIR, TARGET_CHANNEL, test_size=0.2):
    FEATURE_IDXS = [
        0,      # sum
        4,      # rate
        5, 6, 7,
        8, 9, 10,
        11, 12, 13,
        14, 15, 16,
        17, 18
    ] #skipped indexes are just metadata not crucial to this process
    channel_data = {}
    lengths = []
    for ch in range(1, 15):
        path = os.path.join(DATA_DIR, f"channel_{ch}_good_data_aggregated.txt")
        data = np.loadtxt(path, skiprows=1)
        channel_data[ch] = data
        lengths.append(len(data))

    min_len = min(lengths)
    print("Using common length:", min_len)
    X_list = []
    y_list = []

    for i in range(min_len):
        x_row = []
        for ch in range(1, 15):
            features = channel_data[ch][i, FEATURE_IDXS]
            if ch == TARGET_CHANNEL:
                y_row = features
            else:
                x_row.extend(features)
        X_list.append(x_row)
        y_list.append(y_row)

    X = np.array(X_list)
    y = np.array(y_list)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

#################################################################
############################# ANALYSIS ##########################
#################################################################

def interpolate_trapezoid(counts, mid_times, dim):

    f = interp1d(mid_times, counts, kind='linear')
    grid_time = np.linspace(mid_times.min(), mid_times.max(), dim)
    interpolated_counts = f(grid_time)

    mvp = trapz(interpolated_counts, grid_time)

    return grid_time, interpolated_counts, mvp

def standardize(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def run_pca(X):
    pca = PCA()
    Z = pca.fit_transform(X)
    return pca, Z

#################################################################
############################# PLOTS #############################
#################################################################

def generate_and_save_training_and_test_plot_sets(TARGET_CHANNEL, output_path, FEATURE_NAMES, scaler_y, y_test, y_train, y_test_pred, y_train_pred):
    y_train_true_phys = scaler_y.inverse_transform(y_train)
    y_train_pred_phys = scaler_y.inverse_transform(y_train_pred)
    y_test_true_phys  = scaler_y.inverse_transform(y_test)
    y_test_pred_phys  = scaler_y.inverse_transform(y_test_pred)
    if TARGET_CHANNEL <= 6:
        channel_dir = f"Hi_{TARGET_CHANNEL}"
    else:
        channel_dir = f"Lo_{TARGET_CHANNEL}"
    save_dir = os.path.join(output_path, channel_dir)
    os.makedirs(save_dir, exist_ok=True)
    #trainig plots
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()
    for i, ax in enumerate(axes[:len(FEATURE_NAMES)]):
        ax.scatter(y_train_true_phys[:, i], y_train_pred_phys[:, i], s=8, alpha=0.5)
        lims = [min(y_train_true_phys[:, i].min(), y_train_pred_phys[:, i].min()), max(y_train_true_phys[:, i].max(), y_train_pred_phys[:, i].max())]
        ax.plot(lims, lims, "--", color="orange")
        ax.set_title(FEATURE_NAMES[i])
        ax.grid(True)

    plt.suptitle(f"y_pred vs y_true (TRAIN) – channel {TARGET_CHANNEL}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_plot_set_channel_{TARGET_CHANNEL}.png"))
    # plt.show()
    plt.close()
    # test plots
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()
    for i, ax in enumerate(axes[:len(FEATURE_NAMES)]):
        ax.scatter(y_test_true_phys[:, i], y_test_pred_phys[:, i], s=8, alpha=0.5)
        lims = [min(y_test_true_phys[:, i].min(), y_test_pred_phys[:, i].min()), max(y_test_true_phys[:, i].max(), y_test_pred_phys[:, i].max())]
        ax.plot(lims, lims, "--", color="orange")
        ax.set_title(FEATURE_NAMES[i])
        ax.grid(True)

    plt.suptitle(f"y_pred vs y_true (TEST) – channel {TARGET_CHANNEL}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_plot_set_channel_{TARGET_CHANNEL}.png"))
    # plt.show()
    plt.close()

def plot_pred_vs_true(y_true, y_pred, title, filename, label_x='Reference values', label_y='Predicted values', margin=0.03):
    plt.figure(figsize=(16, 9))
    plt.scatter(y_true, y_pred, s=5, alpha=0.4)
    x_min, x_max = y_true.min(), y_true.max()
    y_min, y_max = y_pred.min(), y_pred.max()
    dx = x_max - x_min
    dy = y_max - y_min
    plt.xlim(x_min - margin * dx, x_max + margin * dx)
    plt.ylim(y_min - margin * dy, y_max + margin * dy)
    xx = np.linspace(x_min, x_max, 200)
    plt.plot(xx, xx, '--', color='red', linewidth=2, label='Ideal prediction (y = x)')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_loss_curves(model, out_path, show = False):
    plt.figure(figsize=(10, 6))
    plt.plot(
        model.loss_curve_,
        label="Training loss (MSE)",
        color="blue"
    )
    if (
        hasattr(model, "validation_scores_")
        and model.validation_scores_ is not None
        and len(model.validation_scores_) > 0
    ):
        plt.plot(
            model.validation_scores_,
            label="Validation score (R²)",
            color="orange"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Score")
    plt.title("MLPRegressor training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()
def plot_rate_histograms(rates, title_prefix):
    n_ch = rates.shape[1]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    for ch in range(n_ch):
        axes[ch].hist(rates[:, ch], bins=50, alpha=0.7)
        axes[ch].set_title(f"Channel {ch+1}")
        axes[ch].set_xlabel("Rate")
        axes[ch].set_ylabel("Counts")

    for ax in axes[n_ch:]:
        ax.axis("off")

    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(rates, title, channel_labels=None):
    """
    rates : np.ndarray (N, n_channels)
    """

    corr = np.corrcoef(rates.T)

    if channel_labels is None:
        channel_labels = [f"Ch {i+1}" for i in range(rates.shape[1])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        xticklabels=channel_labels,
        yticklabels=channel_labels,
        square=True,
        cbar_kws={"label": "Pearson r"}
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_explained_variance(pca, title):
    cum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(6, 4))
    plt.plot(cum, marker="o")
    plt.axhline(0.9, linestyle="--", color="gray", label="90%")
    plt.axhline(0.8, linestyle="--", color="lightgray", label="80%")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_projection(Z, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_loadings(pca, pc_idx, title):
    loadings = pca.components_[pc_idx]

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(loadings)+1), loadings)
    plt.xlabel("Channel")
    plt.ylabel("Loading")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#################################################################
############################ HELPERS ############################
#################################################################

def print_and_save_metrics(TARGET_CHANNEL, output_path, y_test, y_train, y_test_pred, y_train_pred, scaler_y):
    # SCALED METRICS
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test   = mean_absolute_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train  = mean_absolute_error(y_train, y_train_pred)

    std_test  = np.std(y_test)
    std_train = np.std(y_train)

    rmse_std_test  = rmse_test / std_test if std_test > 0 else np.inf
    rmse_std_train = rmse_train / std_train if std_train > 0 else np.inf

    # INVERSE SCALING
    y_train_true_phys = scaler_y.inverse_transform(y_train)
    y_train_pred_phys = scaler_y.inverse_transform(y_train_pred)
    y_test_true_phys  = scaler_y.inverse_transform(y_test)
    y_test_pred_phys  = scaler_y.inverse_transform(y_test_pred)

    # PHYSICAL METRICS
    rmse_phys_test  = np.sqrt(mean_squared_error(y_test_true_phys, y_test_pred_phys))
    mae_phys_test   = mean_absolute_error(y_test_true_phys, y_test_pred_phys)
    rmse_phys_train = np.sqrt(mean_squared_error(y_train_true_phys, y_train_pred_phys))
    mae_phys_train  = mean_absolute_error(y_train_true_phys, y_train_pred_phys)

    std_phys_test  = np.std(y_test_true_phys)
    std_phys_train = np.std(y_train_true_phys)

    rmse_std_phys_test  = rmse_phys_test / std_phys_test if std_phys_test > 0 else np.inf
    rmse_std_phys_train = rmse_phys_train / std_phys_train if std_phys_train > 0 else np.inf

    # RESULTS -- SUMMARY
    results_text = (
        f"############# TARGET CHANNEL = {TARGET_CHANNEL} #############\n"
        f"==== TRAINING ====\n"
        f"RMSE (global): {rmse_train:.4f}\n"
        f"MAE  (global): {mae_train:.4f}\n"
        f"RMSE/STD (global): {rmse_std_train:.4f}\n"
        f"RMSE (physical units): {rmse_phys_train:.4f}\n"
        f"MAE  (physical units): {mae_phys_train:.4f}\n"
        f"RMSE/STD (physical units): {rmse_std_phys_train:.4f}\n"
        f"==== TEST ====\n"
        f"RMSE (global): {rmse_test:.4f}\n"
        f"MAE  (global): {mae_test:.4f}\n"
        f"RMSE/STD (global): {rmse_std_test:.4f}\n"
        f"RMSE (physical units): {rmse_phys_test:.4f}\n"
        f"MAE  (physical units): {mae_phys_test:.4f}\n"
        f"RMSE/STD (physical units): {rmse_std_phys_test:.4f}\n"
    )
    print(results_text)
    output_file = os.path.join(output_path, f"test_and_training_metrics.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(results_text)