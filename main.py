from IBEX_Module.TensorAnalyzer import ChannelAnalyzer
from IBEX_Module.TensorCreator import TensorCreator
from IBEX_Module.FileMerger import FileMerger
from IBEX_Module.IBEX_NN import RateAutoencoder
import random
import IBEX_Module.IBEX_NN as IBEX_NN
import torch
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

if __name__ == "__main__":
    print("Script started...")
    #############################################################################################################################################################################################
    #################################################################################### Creating Tensors #######################################################################################
    #############################################################################################################################################################################################

    tensor_creator = TensorCreator("MainConfig.yml")
    tensor_creator._set_creation_params("LoGoodTimes.txt", "lode", "./output/BeforePerigeeChange/Lo_Data/lo_hex", '2009A','2011A', 'o0127', 'o0011')
    tensor_creator.init_channel_tensors(True)
    tensor_creator._set_creation_params("HiCullGoodTimes.txt", "hide", "./output/BeforePerigeeChange/Hi_data/hi_hex", '2009A','2011A','o0127', 'o0011')
    tensor_creator.init_channel_tensors(True)
    print('Channel tensors before major orbital change have been saved')

    #############################################################################################################################################################################################

    tensor_creator._set_creation_params("LoGoodTimes.txt", "lode", "./output/AfterPerigeeChange/Lo_Data/lo_hex", '2011A', '2019B','o0471b', 'o0130a')
    tensor_creator.init_channel_tensors(True)
    tensor_creator._set_creation_params("HiCullGoodTimes.txt", "hide", "./output/AfterPerigeeChange/Hi_data/hi_hex", '2011A', '2019B', 'o0471b', 'o0130a')
    tensor_creator.init_channel_tensors(True)
    print('Channel tensors after major orbital change have been saved')

    ############################################################################################################################################################################################
    ########################################################################## Calculating Detected ENA (sums/timedelta) #######################################################################
    ############################################################################################################################################################################################

    analyzer = ChannelAnalyzer("MainConfig.yml")
    analyzer.init_analyzer_tensors('hi_hex_channel','lo_hex_channel', option="aggregate_all_physical_features")

    ############################################################################################################################################################################################
    ####################################################################### Evaluation using Feed-Forward Neural Network ########################################################################
    ############################################################################################################################################################################################
    print("CWD =", os.getcwd())
    IBEX_NN.init_dataset_preparation_pipeline("channel_analyzer_out", "aggregated_data_for_autoencoder")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(SEED)
        device = "cuda"
    ######  BeforePerigeeChange ######
    Xb, scaler_b = IBEX_NN.load_autoencoder_data(
        "aggregated_data_for_autoencoder/BeforePerigeeChange"
    )
    print("Input shape:", Xb.shape)
    print("Mean:", Xb.mean(axis=0))
    print("Std:", Xb.std(axis=0))
    train_loader, val_loader = IBEX_NN.make_dataloaders(Xb)
    model = RateAutoencoder(input_dim=14, latent_dim=4)
    history = IBEX_NN.train_autoencoder(
        model,
        train_loader,
        val_loader,
        n_epochs=200,
        lr=1e-3,
        device=device
    )
    IBEX_NN.plot_training_history(
        history,
        "AE training – BeforePerigeeChange (latent_dim=4)"
    )
    print(
        "Reconstruction MSE:",
        IBEX_NN.reconstruction_mse(model, Xb, device=device)
    )
    IBEX_NN.plot_per_channel_mse(
        IBEX_NN.per_channel_mse(model, Xb, device=device),
        "Per-channel MSE – BeforePerigeeChange"
    )
    ######  AfterPerigeeChange ######
    Xb, scaler_b = IBEX_NN.load_autoencoder_data(
        "aggregated_data_for_autoencoder/AfterPerigeeChange"
    )
    print("Input shape:", Xb.shape)
    print("Mean:", Xb.mean(axis=0))
    print("Std:", Xb.std(axis=0))
    train_loader, val_loader = IBEX_NN.make_dataloaders(Xb)
    model = RateAutoencoder(input_dim=14, latent_dim=4)
    history = IBEX_NN.train_autoencoder(
        model,
        train_loader,
        val_loader,
        n_epochs=200,
        lr=1e-3,

        device=device
    )
    IBEX_NN.plot_training_history(
        history,
        "AE training – AfterPerigeeChange (latent_dim=4)"
    )
    print(
        "Reconstruction MSE:",
        IBEX_NN.reconstruction_mse(model, Xb, device=device)
    )
    IBEX_NN.plot_per_channel_mse(
        IBEX_NN.per_channel_mse(model, Xb, device=device),
        "Per-channel MSE – AfterPerigeeChange"
    )