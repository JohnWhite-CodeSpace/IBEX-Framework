from IBEX_Module.TensorAnalyzer import ChannelAnalyzer
from IBEX_Module.TensorCreator import TensorCreator
from IBEX_Module.FileMerger import FileMerger
from IBEX_Module.IBEX_NN import RateAutoencoder
import random
import IBEX_Module.IBEX_NN as IBEX_NN
import IBEX_Module.IBEX_RegAutoencoder as IRAE
import torch
import os
import numpy as np

if __name__ == "__main__":
    print("Script started...")
    print("CWD =", os.getcwd())
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
    ######################################################################### Evaluation using RateAutoencoder Network #########################################################################
    ############################################################################################################################################################################################
    IBEX_NN.init_dataset_preparation_pipeline("channel_analyzer_out", "aggregated_data_for_autoencoder")
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

    ############################################################################################################################################################################################
    ######################################################################### Evaluation using RateAutoencoder Network #########################################################################
    ############################################################################################################################################################################################
    TARGET_CHANNEL = 5
    FEATURE_NAMES = [
        "sum", "rate",
        "cos_RA", "sin_RA", "R_RA",
        "cos_phase", "sin_phase", "R_phase"
        , "mean_X", "mean_Y", "mean_Z",
        "std_X", "std_Y", "std_Z",
        "mean_R", "std_R"
    ]

    X_list, y_list = IRAE.load_aggregated_data("channel_analyzer_out_aggregated/AfterPerigeeChange", TARGET_CHANNEL)
    train_loader, val_loader, test_loader, scaler_X, scaler_y = IRAE.build_dataloaders(X_list, y_list, batch_size=256, test_size=0.2, val_size=0.1)
    model = IRAE.IBEX_RegAutoencoder(input_dim=224, dec_output_dim=224, reg_output_dim=16, latent_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, history, metrics = IRAE.train_reg_autoencoder(
        model,
        train_loader,
        val_loader,
        test_loader,
        scaler_y=scaler_y,
        optimizer=optimizer,
        epochs=500,
        lambda_reg=0.1,
        device=device
    )

    y_train_true, y_train_pred = IRAE.get_predictions(model, train_loader, device)
    y_train_true_phys = scaler_y.inverse_transform(y_train_true)
    y_train_pred_phys = scaler_y.inverse_transform(y_train_pred)

    y_test_true, y_test_pred = IRAE.get_predictions(model, test_loader, device)
    y_test_true_phys = scaler_y.inverse_transform(y_test_true)
    y_test_pred_phys = scaler_y.inverse_transform(y_test_pred)

    IRAE.plot_pred_vs_true_grid_4x4(
        y_train_true_phys,
        y_train_pred_phys,
        FEATURE_NAMES,
        title=f"y_pred vs y_true (TRAIN) – channel {TARGET_CHANNEL}"
    )
    IRAE.plot_pred_vs_true_grid_4x4(
        y_test_true_phys,
        y_test_pred_phys,
        FEATURE_NAMES,
        title=f"y_pred vs y_true (TEST) – channel {TARGET_CHANNEL}"
    )

    IRAE.plot_training_curves(history)