import torch
import numpy as np
import os
import yaml
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

__author__ = "Jan Biały"

class TensorCreator:
    """
    TensorCreator
    =============

    Class responsible for constructing structured PyTorch tensors from raw IBEX
    measurement data files.

    The class traverses the IBEX raw data directory tree, loads event-level data
    for individual energy channels, applies basic preprocessing steps
    (including optional hexadecimal value translation), and combines the data
    into channel-specific tensors saved in ``.pt`` format.

    Tensor creation can be performed either for continuous orbital ranges or
    for datasets split around the perigee change of the IBEX spacecraft.

    """
    def __init__(self, config: str):
        """
        Initialize the TensorCreator object using a YAML configuration file.

        The configuration file specifies paths to raw IBEX data directories,
        channel processing parameters, and preprocessing options.

        Parameters
        ----------
        config : str
            Path to the YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        Exception
            If an error occurs during configuration parsing.
        """
        try:
            with open(config, 'r') as config_file:
                self.cfg = yaml.load(config_file, Loader=yaml.FullLoader)

            self.path = self.cfg["TensorCreator"]["FileParams"]["path_to_main_dir"]
            self.start_dir = os.path.join(self.path, self.cfg["TensorCreator"]["CreatorParams"]["start_folder"])
            self.end_dir = os.path.join(self.path, self.cfg["TensorCreator"]["CreatorParams"]["end_folder"])
            self.translate_hex = self.cfg["TensorCreator"]["CreatorParams"]["translate_hex"]

        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Error: No such file as {config}. Exception: {ex}")
        except Exception as ex:
            raise Exception(f"Error: {repr(ex)}")

    def _set_creation_params(self, instruction_file: str, file_type: str, savefile_prefix: str, start_dir: str | None = None, end_dir: str | None = None, end_subfolder: str | None = None,start_subfolder: str | None = None) -> None:
        """
        Set internal parameters controlling tensor creation.

        This method configures the instruction file, data file type, output
        filename prefix, and optional directory boundaries used during
        tensor generation.

        Parameters
        ----------
        instruction_file : str
            Name of the IBEX instruction file defining valid data intervals.
        file_type : str
            Data file identifier (e.g., ``'hide'`` for IBEX-Hi or ``'lode'`` for IBEX-Lo).
        savefile_prefix : str
            Prefix for generated tensor filenames.
        start_dir : str, optional
            Starting directory for data traversal.
        end_dir : str, optional
            Ending directory for data traversal.
        start_subfolder : str, optional
            Starting subfolder for perigee-change-based processing.
        end_subfolder : str, optional
            Ending subfolder for perigee-change-based processing.

        Returns
        -------
        None
        """
        self.instruction_file = instruction_file
        self.file_type = file_type
        self.savefile_prefix = savefile_prefix
        if end_subfolder:
            self.end_subfolder = end_subfolder
            self.start_subfolder = start_subfolder
        if start_dir and end_dir:
            self.start_dir = start_dir
            self.end_dir = end_dir

    def init_channel_tensors(self, perigee_change: bool = False)-> None:
        """
        Initialize multithreaded tensor creation for all energy channels.

        The method spawns a separate worker thread for each energy channel and
        generates channel-specific tensors according to the configured
        instruction file and directory boundaries.

        Parameters
        ----------
        perigee_change : bool, optional
            If True, tensor creation is performed across subfolders spanning
            the perigee change of the IBEX spacecraft. If False, data are processed
            within a continuous directory range.

        Raises
        ------
        ValueError
            If an unsupported instruction file is specified.

        Returns
        -------
        None
        """
        channel_num = 6 if self.instruction_file == "HiCullGoodTimes.txt" else 8 if self.instruction_file == "LoGoodTimes.txt" else None
        if not channel_num:
            raise ValueError("Incorrect instruction file. Aborting...")

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._init_channel_tensor, i, perigee_change) for i in range(1, channel_num + 1)]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during processing: {e}")

    def _init_channel_tensor(self, channel_index: int, perigee_change: bool = False)-> None:
        """
        Generate a tensor for a single IBEX energy channel.

        The method traverses the raw data directory tree, identifies files
        corresponding to the specified channel, loads and preprocesses the data,
        and aggregates all valid records into a single PyTorch tensor.

        Processing can be performed either within a continuous directory range
        or across subfolders corresponding to the perigee change of the IBEX
        spacecraft.

        Parameters
        ----------
        channel_index : int
            Energy channel index (1–6 for IBEX-Hi, 1–8 for IBEX-Lo).
        perigee_change : bool, optional
            Controls whether perigee-change-aware directory traversal is used.

        Returns
        -------
        None
        """
        batch_data_list = []
        channel_file_regex = f"{self.file_type}-{channel_index}"
        save_path = f"{self.savefile_prefix}_channel_{channel_index}.pt"
        base_path = self.path

        if not perigee_change:
            all_folders = sorted([
                os.path.join(base_path, d)
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ])

            try:
                start_i = all_folders.index(os.path.join(base_path, self.start_dir))
                end_i = all_folders.index(os.path.join(base_path, self.end_dir))
            except ValueError as e:
                print(f"There is no such directories as {self.start_dir} or {self.end_dir} in provided path: {e}")
                return

            target_folders = all_folders[start_i:end_i + 1]

            for folder in target_folders:
                for root, _, files in os.walk(folder):
                    data_list = self._process_files(
                        [f for f in files if channel_file_regex in f], root
                    )
                    batch_data_list.extend(data_list)

        else:
            all_subfolders = []
            for main_folder in sorted(os.listdir(base_path)):
                main_path = os.path.join(base_path, main_folder)
                if not os.path.isdir(main_path):
                    continue
                for subfolder in sorted(os.listdir(main_path)):
                    sub_path = os.path.join(main_path, subfolder)
                    if os.path.isdir(sub_path):
                        all_subfolders.append(sub_path)
            all_subfolders = [os.path.normpath(p) for p in all_subfolders]
            try:
                print(f"All subfolders found: {all_subfolders}")
                start_path = os.path.normpath(os.path.join(base_path, self.start_dir, self.start_subfolder))
                end_path = os.path.normpath(os.path.join(base_path, self.end_dir, self.end_subfolder))
                start_i = all_subfolders.index(start_path)
                end_i = all_subfolders.index(end_path)
            except ValueError as e:
                print(f"There is no such subdirectories as {self.start_subfolder} or {self.end_subfolder} in provided path: {e}")
                return

            target_subfolders = all_subfolders[start_i:end_i + 1]

            for subfolder in target_subfolders:
                for root, _, files in os.walk(subfolder):
                    data_list = self._process_files(
                        [f for f in files if channel_file_regex in f], root
                    )
                    batch_data_list.extend(data_list)

        if batch_data_list:
            combined_data = np.vstack(batch_data_list).astype(np.float32)
            torch.save(torch.tensor(combined_data), save_path)
            print(f"Saved tensor with shape {combined_data.shape} to {save_path}")
            del combined_data, batch_data_list
            gc.collect()
        else:
            print(f"No data from channel {channel_index}")

    def _process_files(self, files: list[str], root: str)-> list[np.ndarray]:
        """
        Load and preprocess raw IBEX data files for a single directory.

        The method reads data files as strings, removes or converts hexadecimal
        values, and converts the data to floating-point format.

        Parameters
        ----------
        files : list of str
            List of filenames to be processed.
        root : str
            Root directory containing the files.

        Returns
        -------
        list of np.ndarray
            List of preprocessed data arrays.
        """
        data_list = []
        for file in files:
            if self.file_type in file:
                file_path = os.path.join(root, file)
                text = np.loadtxt(file_path, dtype='str')
                text = self._remove_or_convert_hex_flags(text)
                data_list.append(text.astype(np.float32))
        gc.collect()
        return data_list

    def _remove_or_convert_hex_flags(self, data_list: np.ndarray) -> np.ndarray:
        """
        Remove or convert hexadecimal flag values in raw IBEX data arrays.

        Depending on the configuration, hexadecimal values are either translated
        into integers or replaced with zeros.

        Parameters
        ----------
        data_list : np.ndarray
            Raw data array containing hexadecimal flag values as strings.

        Returns
        -------
        np.ndarray
            Array with hexadecimal values converted or removed.
        """
        if self.translate_hex:
            data_list[:, 3] = np.vectorize(lambda x: int(x, 16))(data_list[:, 3])
            data_list[:, 4] = np.vectorize(lambda x: int(x, 16))(data_list[:, 4])
            data_list[:, 6] = 0
        else:
            data_list[:, 3] = 0
            data_list[:, 4] = 0
            data_list[:, 6] = 0
        gc.collect()
        return data_list