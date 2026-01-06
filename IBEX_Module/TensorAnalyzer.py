import torch
import numpy as np
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

__author__ = "Jan Biały"

class ChannelAnalyzer:
    """
    ChannelAnalyzer
    ===============

    Class responsible for preprocessing IBEX ENA data based on high-quality acquisition
    time intervals specified in instruction files (HiCullGoodTimes.txt and LoGoodTimes.txt).

    The class provides three main data preprocessing modes:

    1. Extraction of raw ENA events satisfying quality criteria and saving them as PyTorch tensors.
    2. Calculation of good data sums (total number of ENA detections per valid time interval).
    3. Aggregation of ENA detection features by computing physically meaningful statistical
       descriptors (means, circular means, and standard deviations).

    The preprocessing is performed independently for each IBEX-Hi and IBEX-Lo energy channel.

    Notes
    -----
    Some methods of this class are marked as deprecated and preserved only as
    legacy artefacts from earlier versions of the analysis pipeline. They are
    not used in the current implementation but remain available for reference
    and reproducibility of previous results.

    ------
    """
    def __init__(self, config: str = "MainConfig.yml"):
        """
        Initialize the ChannelAnalyzer object.

        The class requires a configuration file in YAML format, which specifies
        paths to input tensor directories, instruction files, and output directories.

        Parameters
        ----------
        config : str, optional
            Path to the YAML configuration file. Default is ``"MainConfig.yml"``.

        Raises
        ------
        FileNotFoundError
            If the provided configuration file does not exist.
        """
        try:
            with open(config, 'r') as config_file:
                self.cfg = yaml.load(config_file, Loader=yaml.FullLoader)
            self.hi_tensor_directory = self.cfg["ChannelAnalyzer"]["hi_tensor_directory"]
            self.lo_tensor_directory = self.cfg["ChannelAnalyzer"]["lo_tensor_directory"]
            self.output_dir = self.cfg["ChannelAnalyzer"]["output_dir"]
            os.makedirs(self.output_dir, exist_ok=True)
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Error: No such file as {config}. Exception: {ex}")

    def init_analyzer_tensors(self, hi_name: str ='hi_hex_channel', lo_name: str ='lo_hex_channel', option: str = "save_good_data_sums")-> None:
        """
        Initialize multithreaded preprocessing of all IBEX energy channels.

        Depending on the selected option, the method performs one of the supported
        preprocessing procedures independently for each channel using a thread pool.

        Parameters
        ----------
        hi_name : str, optional
            Filename prefix for IBEX-Hi channel tensor files.
        lo_name : str, optional
            Filename prefix for IBEX-Lo channel tensor files.
        option : str, optional
            Type of preprocessing to perform. Supported values are:

            - ``"save_raw_good_data"`` — extract and save raw filtered ENA events.
            - ``"save_good_data_sums"`` — compute and save ENA detection sums.
            - ``"aggregate_all_physical_features"`` — aggregate ENA features for global analysis.

        Raises
        ------
        Exception
            If an unsupported option value is provided.
        """
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    self._process_channel,
                    i, hi_name, lo_name, option
                )
                for i in range(1, 15)
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during processing channel: {e}")

    def _process_channel(self, i: int, hi_name:str , lo_name: str, option: str)-> None:
        """
        Process a single IBEX energy channel according to the selected preprocessing option.

        The method loads the corresponding tensor file for the given channel index,
        selects the appropriate instruction file (IBEX-Hi or IBEX-Lo), and applies
        one of the supported preprocessing procedures:

        - extraction of raw ENA events,
        - computation of good data sums,
        - aggregation of physical features for global analysis.

        The results are saved to disk in the output directory specified in the
        configuration file.

        Parameters
        ----------
        i : int
            Global channel index (1–14). Channels 1–6 correspond to IBEX-Hi,
            while channels 7–14 correspond to IBEX-Lo.
        hi_name : str
            Filename prefix for IBEX-Hi tensor files.
        lo_name : str
            Filename prefix for IBEX-Lo tensor files.
        option : str
            Preprocessing option to be applied. See ``init_analyzer_tensors``
            for the list of supported values.

        Returns
        -------
        None
        """
        tensor_path = (
            os.path.join(self.hi_tensor_directory, f"{hi_name}_{i}.pt")
            if i <= 6 else
            os.path.join(self.lo_tensor_directory, f"{lo_name}_{i - 6}.pt")
        )

        instruction_file = (
            self.cfg["ChannelAnalyzer"]["hi_instruction_file"]
            if i <= 6 else
            self.cfg["ChannelAnalyzer"]["lo_instruction_file"]
        )

        if not os.path.exists(tensor_path):
            print(f"Tensor file not found: {tensor_path}. Skipping channel {i}.")
            return

        try:
            print(f"Loading {tensor_path}")
            tensor = torch.load(tensor_path).numpy()
        except Exception as e:
            print(f"Error loading tensor for channel {i}: {e}. Skipping...")
            return

        is_hi_channel = i <= 6
        if option=="save_raw_good_data":
            good_data = self._extract_good_raw_data(
                tensor, instruction_file, i, is_hi_channel
            )

            output_file = os.path.join(
                self.output_dir, f"channel_{i}_good_raw.pt"
            )

            torch.save(torch.from_numpy(good_data), output_file)
            print(f"Saved RAW good data tensor for channel {i}: shape={good_data.shape}")
            return

        elif  option =="save_good_data_sums":
            good_data_sums = self._calculate_good_data_sums(
                tensor, instruction_file, i, is_hi_channel
            )

            output_file = os.path.join(
                self.output_dir, f"channel_{i}_good_data.txt"
            )

            try:
                header = "sum time_delta start_time end_time"
                np.savetxt(output_file, good_data_sums, fmt="%.6f", header=header, comments="")
                print(f"Saved good data sums for channel {i} to {output_file}")
            except Exception as e:
                print(f"Error saving good data sums for channel {i}: {e}")

        elif option == "aggregate_all_physical_features":
            good_data_aggregated = self._aggregate_data_for_global_analysis(
                tensor, instruction_file, i, is_hi_channel)
            output_file = os.path.join(self.output_dir, f"channel_{i}_good_data_aggregated.pt")
            try:
                header = "sum time_delta start_time end_time rate mean_cos_RA mean_sin_RA R_RA mean_cos_phase mean_sin_phase R_phase mean_X_RE, mean_Y_RE, mean_Z_RE, std_X_RE, std_Y_RE, std_Z_RE, mean_R, std_R"
                np.savetxt(output_file, good_data_aggregated, fmt="%.6f", header=header, comments="")
                print(f"Saved good data aggregated for channel {i} to {output_file}")
            except Exception as e:
                print(f"Error saving good data aggregated for channel {i}: {e}")
        else:
            raise Exception(f"Unknown option: {option}")


    def _extract_good_raw_data(self, tensor: np.ndarray, instruction_file: str, channel_index: int, is_hi_channel: bool)-> np.ndarray:
        """
        Extract raw ENA events that satisfy quality criteria defined in the instruction file.

        The method filters the input tensor based on valid time intervals and
        phase/channel availability conditions specified in the corresponding
        instruction file.

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).

        Returns
        -------
        np.ndarray
            Array containing filtered ENA events. If no valid events are found,
            an empty array is returned.
        """
        dtype = self._get_dtype(is_hi_channel, channel_index)

        instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)

        good_intervals = self._extract_good_data_intervals(
            instruction_data, channel_index
        )

        good_rows = []

        for interval in good_intervals:
            start_time = interval['start_time']
            end_time = interval['end_time']

            mask = (tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time)
            valid = tensor[mask]

            if valid.size > 0:
                good_rows.append(valid)

        if not good_rows:
            return np.empty((0, tensor.shape[1]))

        return np.vstack(good_rows)

    def _calculate_good_data_sums(self, tensor: np.ndarray, instruction_file:str, channel_index: int, is_hi_channel: bool)-> np.ndarray:
        """
        Compute sums of valid ENA detection events for each good time interval.

        For each time interval satisfying quality criteria, the method calculates
        the total number of ENA detections and associates it with the corresponding
        time span.

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).

        Returns
        -------
        np.ndarray
            Array of shape (N, 4) containing:
            [sum, time_delta, start_time, end_time] for each valid interval.
        """
        dtype = self._get_dtype(is_hi_channel, channel_index)
        try:
            instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)
        except Exception as e:
            print(f"Error reading instruction file {instruction_file}: {e}")
            return []

        good_data_intervals = self._extract_good_data_intervals(instruction_data, channel_index)
        good_data_sums = []
        for interval in good_data_intervals:
            start_time = interval['start_time']
            end_time = interval['end_time']
            time_delta = end_time - start_time
            valid_data = tensor[(tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time)]
            if valid_data.size > 0:
                sum_valid_data = valid_data[:, 5].sum().item()
                good_data_sums.append([sum_valid_data, time_delta, start_time, end_time])

        return np.array(good_data_sums)

    def _aggregate_data_for_global_analysis(self, tensor: np.ndarray, instruction_file: str, channel_index: int, is_hi_channel: bool)-> np.ndarray:
        """
        Aggregate ENA detection data into physically meaningful statistical features.

        For each valid time interval, the method computes aggregated descriptors
        such as sums, rates, circular means of angular quantities, and standard
        deviations of spacecraft position parameters.

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).

        Returns
        -------
        np.ndarray
            Aggregated feature matrix for global analysis.
        """
        dtype = self._get_dtype(is_hi_channel, channel_index)
        try:
            instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)
        except Exception as e:
            print(f"Error reading instruction file {instruction_file}: {e}")
            return []
        good_data_intervals = self._extract_good_data_intervals(instruction_data, channel_index)
        good_data_aggregated= []
        for interval in good_data_intervals:
            start_time = interval['start_time']
            end_time = interval['end_time']
            time_delta = end_time - start_time
            valid_data = tensor[(tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time)]
            if valid_data.size > 0:
                sum_valid_data = valid_data[:, 5].sum().item()
                rate = sum_valid_data / time_delta
                # direction of incoming ENA particle
                RA = np.deg2rad(valid_data[:, 1])  # if RA in deg
                mean_cos_RA = np.mean(np.cos(RA))
                mean_sin_RA = np.mean(np.sin(RA))
                R_RA = np.sqrt(mean_cos_RA ** 2 + mean_sin_RA ** 2)
                # phase of the channel lens
                phase = valid_data[:, 7]
                mean_cos_phase = np.mean(np.cos(2 * np.pi * phase))
                mean_sin_phase = np.mean(np.sin(2 * np.pi * phase))
                R_phase = np.sqrt(mean_cos_phase ** 2 + mean_sin_phase ** 2)
                # IBEX sattelite location relative to Earth (X, Y, Z)
                X = valid_data[:, 8]
                Y = valid_data[:, 9]
                Z = valid_data[:, 10]
                mean_X_RE = np.mean(X)
                mean_Y_RE = np.mean(Y)
                mean_Z_RE = np.mean(Z)
                std_X_RE = np.std(X)
                std_Y_RE = np.std(Y)
                std_Z_RE = np.std(Z)
                R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                mean_R = np.mean(R)
                std_R = np.std(R)

                good_data_aggregated.append([sum_valid_data, time_delta, start_time,
                                             end_time, rate, mean_cos_RA,
                                             mean_sin_RA, R_RA, mean_cos_phase,
                                             mean_sin_phase, R_phase, mean_X_RE,
                                             mean_Y_RE, mean_Z_RE, std_X_RE,
                                             std_Y_RE, std_Z_RE, mean_R, std_R])
        return np.array(good_data_aggregated)

    def _extract_good_data_intervals(self, instruction_data, channel_index: int)-> List[Dict[str, float]]:
        """
        Extract valid ENA acquisition time intervals from instruction data.

        A time interval is considered valid if the full phase range (0–59) is covered
        and the corresponding channel availability flag is set to 1.

        Parameters
        ----------
        instruction_data : np.ndarray
            Structured array loaded from the instruction file.
        channel_index : int
            Index of the analyzed energy channel.

        Returns
        -------
        list of dict
            List of dictionaries with keys ``'start_time'`` and ``'end_time'``.
        """
        phase_start_col = 3
        phase_end_col = 4
        channel_bool_checker_col = 5 + channel_index
        if channel_bool_checker_col >= 13:
            channel_bool_checker_col = channel_bool_checker_col - 7
        good_data_intervals = []
        for row in instruction_data:
            if row[channel_bool_checker_col] == 1 and row[phase_start_col] == 0 and row[phase_end_col] == 59:
                start_time = row[1]
                end_time = row[2]
                good_data_intervals.append({'start_time': start_time, 'end_time': end_time})
        return good_data_intervals

    def _get_dtype(self, is_hi_channel: bool, channel_index: int):
        """
        dtype separation for each of the IBEX measuring devices (IBEX-Hi, IBEX-Lo).
        :param is_hi_channel:
        :param channel_index: not used - artefact from previous version of this class
        :return:
        """
        if is_hi_channel:
            return [('orbit', 'i4'), ('start_time', 'f8'), ('end_time', 'f8'), ('phase_start', 'i4'),
                    ('phase_end', 'i4'), ('dataset', 'U2'), ('channel_1', 'i4'), ('channel_2', 'i4'),
                    ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4')]
        else:
            return [('orbit', 'i4'), ('start_time', 'f8'), ('end_time', 'f8'), ('phase_start', 'i4'),
                    ('phase_end', 'i4'), ('dataset', 'U2'), ('channel_1', 'i4'), ('channel_2', 'i4'),
                    ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4'),
                    ('channel_7', 'i4'), ('channel_8', 'i4')]

############################################################  depracated functions  #################################################################################################
    def pearson_correlation(self, x: np.ndarray, y: np.ndarray)-> float:
        """
        .. deprecated:: 1.1
           This method is deprecated and preserved only as a legacy implementation
           from a previous version of the ChannelAnalyzer class. It is no longer
           used in the current analysis pipeline.

        Compute the Pearson correlation coefficient between two one-dimensional arrays.

        Parameters
        ----------
        x : np.ndarray
            First input array.
        y : np.ndarray
            Second input array.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        if len(x) != len(y):
            min_length = min(len(x), len(y))
            x = x[:min_length]
            y = y[:min_length]

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
        if denominator == 0:
            return 0
        return numerator / denominator

    def mutual_information(self, x: np.ndarray, y: np.ndarray)-> float:
        """
        .. deprecated:: 1.1
           Deprecated legacy method retained for reference purposes. Mutual
           information analysis is no longer part of the current processing workflow.

        Estimate mutual information between two one-dimensional arrays.

        Parameters
        ----------
        x : np.ndarray
            First input array.
        y : np.ndarray
            Second input array.

        Returns
        -------
        float
            Estimated mutual information value.
        """
        if len(x) != len(y):
            min_length = min(len(x), len(y))
            x, y = x[:min_length], y[:min_length]
        x = np.digitize(x, bins=np.histogram_bin_edges(x, bins='auto'))
        y = np.digitize(y, bins=np.histogram_bin_edges(y, bins='auto'))
        return mutual_info_score(x, y)

    def spearman_correlation(self, x, y)-> float:
        """
        .. deprecated:: 1.1
           Legacy method retained for backward compatibility with older analysis
           scripts. Not used in the current version of the pipeline.

        Compute the Spearman rank correlation coefficient.

        Parameters
        ----------
        x : np.ndarray
            First input array.
        y : np.ndarray
            Second input array.

        Returns
        -------
        float
            Spearman rank correlation coefficient.
        """
        if len(x) != len(y):
            min_length = min(len(x), len(y))
            x, y = x[:min_length], y[:min_length]
        stat, _ = spearmanr(x, y)
        return stat

    def weighted_pearson_correlation(self, x: np.ndarray, y: np.ndarray, time_durations_x: np.ndarray, time_durations_y: np.ndarray)-> float:
        """
        .. deprecated:: 1.1
           Deprecated weighted correlation method retained as a historical artefact
           from an earlier analysis approach. It is not used in the current workflow.

        Compute a weighted Pearson correlation coefficient using time durations as weights.

        Parameters
        ----------
        x : np.ndarray
            First input array.
        y : np.ndarray
            Second input array.
        time_durations_x : np.ndarray
            Weights associated with the first array.
        time_durations_y : np.ndarray
            Weights associated with the second array.

        Returns
        -------
        float
            Weighted Pearson correlation coefficient.
        """
        min_length = min(len(x), len(y))

        x = x[:min_length]
        y = y[:min_length]
        time_durations_x = time_durations_x[:min_length]
        time_durations_y = time_durations_y[:min_length]

        weights_x = time_durations_x / np.sum(time_durations_x)
        weights_y = time_durations_y / np.sum(time_durations_y)
        weighted_mean_x = np.sum(x * weights_x)
        weighted_mean_y = np.sum(y * weights_y)
        diff_x = x - weighted_mean_x
        diff_y = y - weighted_mean_y
        weighted_diff_x = diff_x * weights_x
        weighted_diff_y = diff_y * weights_y
        numerator = np.sum(weighted_diff_x * weighted_diff_y)
        denominator_x = np.sum(weighted_diff_x ** 2)
        denominator_y = np.sum(weighted_diff_y ** 2)
        denominator = np.sqrt(denominator_x * denominator_y)

        if denominator != 0:
            return numerator / denominator
        else:
            return 0

    def load_data(self, file_path: str)-> np.ndarray:
        """
        .. deprecated:: 1.1
           Deprecated I/O helper method retained from a previous version of the class.
           File loading is no longer handled by this component.

        Load numerical data from a text file.

        Parameters
        ----------
        file_path : str
            Path to the input file.

        Returns
        -------
        np.ndarray
            Loaded data array, or an empty array in case of an error.
        """
        try:
            data = np.loadtxt(file_path)
            return data
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def analyze(self, method: str, file_path_prefix: str, file_path_suffix: str)-> None:
        """
        .. deprecated:: 1.1
           Deprecated end-to-end analysis routine from an earlier version of the
           ChannelAnalyzer class. This method is retained for archival and reference
           purposes only and is not used in the current data processing pipeline.

        Perform correlation analysis between all energy channels and generate
        correlation matrices and heatmaps.

        Parameters
        ----------
        method : str
            Correlation method identifier (``'pearson'``, ``'weighted_pearson'``,
            ``'spearman'``, or ``'MI'``).
        file_path_prefix : str
            Prefix of input file paths.
        file_path_suffix : str
            Suffix of input file paths.

        Returns
        -------
        None
        """
        data = []
        time_durations = []
        for i in range(1, 15):
            file_path = f"{file_path_prefix}{i}{file_path_suffix}.txt"
            file_data = self.load_data(file_path)
            if len(file_data) > 0:
                data.append(file_data[:, 0])
                time_durations.append(file_data[:, 1])
            else:
                print(f"Could not load data: {file_path}")

        if method == "pearson":
            correlation_matrix = np.zeros((14, 14))
            for i in range(14):
                for j in range(14):
                    correlation_matrix[i, j] = self.pearson_correlation(data[i], data[j])

        elif method == "weighted_pearson":
            correlation_matrix = np.zeros((14, 14))
            for i in range(14):
                for j in range(14):
                    correlation_matrix[i, j] = self.weighted_pearson_correlation(data[i], data[j], time_durations[i], time_durations[j])

        elif method == "spearman":
            correlation_matrix = np.zeros((14, 14))
            for i in range(14):
                for j in range(14):
                    correlation_matrix[i, j] = self.spearman_correlation(data[i], data[j])

        elif method == "MI":
            correlation_matrix = np.zeros((14, 14))
            for i in range(14):
                for j in range(14):
                    correlation_matrix[i, j] = self.mutual_information(data[i], data[j])

        else:
            print("There is no such method.")
            return

        np.savetxt(f"{self.output_dir}/correlation_matrix_{method}.txt", correlation_matrix, delimiter=',', fmt='%.4f')

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.4f', cmap='coolwarm',
                    xticklabels=[f'Plik {i + 1}' for i in range(14)], yticklabels=[f'Plik {i + 1}' for i in range(14)])
        plt.title(f'Macierz współczynników korelacji {method.capitalize()}')
        plt.savefig(f"{self.output_dir}/heatmap_{method}.png")
        plt.show()