import torch
import numpy as np
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

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

    def init_analyzer_tensors(self, hi_name: str ='hi_hex_channel', lo_name: str ='lo_hex_channel', option: str = "save_good_data_sums", no_data_option: str = '')-> None:
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
                    i, hi_name, lo_name, option, no_data_option
                )
                for i in range(1, 15)
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during processing channel: {e}")

    def _process_channel(self, i: int, hi_name:str , lo_name: str, option: str, no_data_option = "")-> None:
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
                tensor, instruction_file, i, is_hi_channel, no_data_option
            )

            output_file = os.path.join(
                self.output_dir, f"channel_{i}_good_raw.pt"
            )

            torch.save(torch.from_numpy(good_data), output_file)
            print(f"Saved RAW good data tensor for channel {i}: shape={good_data.shape}")
            return

        elif  option =="save_good_data_sums":
            good_data_sums = self._calculate_good_data_sums(
                tensor, instruction_file, i, is_hi_channel, no_data_option
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
                tensor, instruction_file, i, is_hi_channel, no_data_option)
            output_file = os.path.join(self.output_dir, f"channel_{i}_good_data_aggregated.pt")
            try:
                header = "sum time_delta start_time end_time rate mean_cos_RA mean_sin_RA R_RA mean_cos_phase mean_sin_phase R_phase mean_X_RE, mean_Y_RE, mean_Z_RE, std_X_RE, std_Y_RE, std_Z_RE, mean_R, std_R"
                np.savetxt(output_file, good_data_aggregated, fmt="%.6f", header=header, comments="")
                print(f"Saved good data aggregated for channel {i} to {output_file}")
            except Exception as e:
                print(f"Error saving good data aggregated for channel {i}: {e}")
        else:
            raise Exception(f"Unknown option: {option}")

    def _extract_good_raw_data(self, tensor: np.ndarray, instruction_file: str, channel_index: int, is_hi_channel: bool, no_data_option: str) -> np.ndarray:
        """
        Extract raw ENA detection events for time intervals in which the analyzed
        energy channel was operational.

        The method uses the instruction file to determine valid acquisition
        time intervals. Only intervals where the channel was active
        (i.e. not marked with ``replace=True``) are considered, because raw ENA
        detection events do not exist when the channel was inactive.

        Time intervals flagged with ``replace=True`` are intentionally skipped at
        the raw-data level. Temporal continuity across channels is preserved later
        during the aggregated-data stage, where inactive-channel intervals are
        represented by placeholder values (e.g. ``-1``).

        This design ensures that:
        - raw event tensors contain only physically meaningful ENA detections,
        - no artificial or duplicated raw events are introduced,
        - temporal alignment across channels is handled consistently at a higher
          level of data processing.

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel. Each row corresponds to
            a single ENA detection event, with time information stored in the first
            column.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals and
            channel availability flags.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).
        no_data_option : str
            Strategy defining how non-operational channel intervals are handled
            in the instruction parsing stage (e.g. replacement marking).

        Returns
        -------
        np.ndarray
            Array containing filtered raw ENA detection events corresponding only
            to time intervals where the channel was operational. If no valid events
            are found, an empty array of shape ``(0, tensor.shape[1])`` is returned.
        """
        dtype = self._get_dtype(is_hi_channel, channel_index)
        instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)
        good_intervals = self._extract_good_data_intervals(instruction_data, channel_index, device="Hi" if is_hi_channel else "Lo", option=no_data_option)
        good_rows = []

        for interval in good_intervals:
            if interval.get("replace", False):
                # kanał nie pracował → brak raw zdarzeń
                continue
            start_time = interval["start_time"]
            end_time = interval["end_time"]
            mask = ((tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time))
            valid = tensor[mask]
            if valid.size > 0:
                good_rows.append(valid)
        if not good_rows:
            return np.empty((0, tensor.shape[1]))

        return np.vstack(good_rows)

    def _calculate_good_data_sums(self, tensor: np.ndarray, instruction_file: str, channel_index: int, is_hi_channel: bool, no_data_option: str) -> np.ndarray:
        """
        Compute sums of ENA detection events for each valid acquisition time interval.

        The method uses the instruction file to identify valid time intervals and
        channel availability conditions. For each interval, the total number of
        detected ENA particles is computed and associated with the corresponding
        time span.

        Time intervals marked with ``replace=True`` indicate periods in which the
        analyzed channel was inactive. In such cases, the sum of ENA detections is
        explicitly set to ``-1`` to denote missing or invalid data while preserving
        temporal continuity across channels.

        If the channel was active but no ENA events are present within a valid
        interval, the sum is set to ``0``, reflecting a physically meaningful
        zero-count observation rather than missing data.

        This distinction allows downstream analyses (e.g. interpolation, machine
        learning, correlation studies) to reliably differentiate between:
        - inactive-channel intervals (``sum = -1``),
        - active-channel intervals with zero detections (``sum = 0``),
        - active-channel intervals with measured ENA counts (``sum > 0``).

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel. Each row corresponds to
            a single ENA detection event, with time information stored in the first
            column.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals and
            channel availability flags.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).
        no_data_option : str
            Strategy defining how inactive-channel intervals are handled during
            instruction parsing (e.g. marking intervals for replacement).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 4)`` containing aggregated ENA detection information
            for each valid time interval in the form:

            ``[sum, time_delta, start_time, end_time]``

            where ``sum = -1`` denotes inactive-channel intervals and ``sum = 0``
            denotes active intervals with no detected ENA events.
        """

        dtype = self._get_dtype(is_hi_channel, channel_index)
        try:
            instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)
        except Exception as e:
            print(f"Error reading instruction file {instruction_file}: {e}")
            return np.empty((0, 4))

        good_data_intervals = self._extract_good_data_intervals(instruction_data, channel_index, device="Hi" if is_hi_channel else "Lo", option=no_data_option)
        good_data_sums = []
        for interval in good_data_intervals:
            start_time = interval["start_time"]
            end_time = interval["end_time"]
            replace = interval.get("replace", False)
            time_delta = end_time - start_time
            # === CASE 1: kanał nieaktywny → sum = -1 ===
            if replace:
                good_data_sums.append([-1.0, time_delta, start_time, end_time])
                continue
            # === CASE 2: normalne liczenie ===
            valid_data = tensor[(tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time)]
            if valid_data.size == 0:
                # brak danych mimo aktywności
                good_data_sums.append([0.0, time_delta, start_time, end_time])
                continue
            sum_valid_data = valid_data[:, 5].sum().item()
            good_data_sums.append([sum_valid_data, time_delta, start_time, end_time])
        return np.asarray(good_data_sums)

    def _aggregate_data_for_global_analysis(self, tensor: np.ndarray, instruction_file: str, channel_index: int, is_hi_channel: bool, no_data_option: str,) -> np.ndarray:
        """
        Aggregate ENA detection data into physically meaningful statistical features
        for global, multi-channel analysis.

        The method uses the instruction file to determine valid acquisition time
        intervals and channel availability conditions. For each interval, a fixed
        set of statistical descriptors is computed from the raw ENA event data,
        including total counts, rates, circular statistics of angular quantities,
        and spacecraft position moments.

        Time intervals marked with ``replace=True`` indicate periods in which the
        analyzed channel was inactive or otherwise unavailable. In such cases,
        all aggregated features for the interval are explicitly set to ``-1``.
        This convention preserves temporal continuity across channels while
        allowing downstream analyses to clearly distinguish missing or invalid
        measurements from physically meaningful zero-valued quantities.

        If the channel was active but no ENA events are present within a valid
        interval, the interval is also filled with ``-1`` values. This ensures a
        consistent feature dimensionality and avoids introducing spurious
        statistics based on empty samples.

        The resulting feature matrix is therefore temporally aligned across all
        channels and devices, enabling robust use in machine learning models,
        correlation studies, and global statistical analyses.

        Parameters
        ----------
        tensor : np.ndarray
            Raw ENA data tensor for the selected channel. Each row corresponds to
            a single ENA detection event, with time information stored in the first
            column.
        instruction_file : str
            Path to the instruction file defining valid acquisition intervals,
            phase coverage, and channel availability flags.
        channel_index : int
            Index of the analyzed energy channel.
        is_hi_channel : bool
            Indicates whether the data correspond to IBEX-Hi (True) or IBEX-Lo (False).
        no_data_option : str
            Strategy defining how inactive-channel intervals are handled during
            instruction parsing (e.g. marking intervals for replacement).

        Returns
        -------
        np.ndarray
            Aggregated feature matrix of shape ``(N, 19)``, where each row corresponds
            to a single acquisition interval and contains the following features:

            ``[sum, time_delta, start_time, end_time, rate,
            mean_cos_RA, mean_sin_RA, R_RA,
            mean_cos_phase, mean_sin_phase, R_phase,
            mean_X_RE, mean_Y_RE, mean_Z_RE,
            std_X_RE, std_Y_RE, std_Z_RE,
            mean_R, std_R]``

            Rows filled with ``-1`` denote intervals in which the channel was inactive
            or the data were not suitable for aggregation.
        """

        dtype = self._get_dtype(is_hi_channel, channel_index)
        try:
            instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)
        except Exception as e:
            print(f"Error reading instruction file {instruction_file}: {e}")
            return np.empty((0, 19))

        good_data_intervals = self._extract_good_data_intervals(instruction_data,channel_index, device="Hi" if is_hi_channel else "Lo", option=no_data_option)
        good_data_aggregated = []
        N_FEATURES = 19
        for interval in good_data_intervals:
            start_time = interval["start_time"]
            end_time = interval["end_time"]
            replace = interval.get("replace", False)
            if replace:
                good_data_aggregated.append([-1.0] * N_FEATURES)
                continue
            time_delta = end_time - start_time
            valid_data = tensor[
                (tensor[:, 0] >= start_time) &
                (tensor[:, 0] <= end_time)
                ]

            if valid_data.size == 0:
                good_data_aggregated.append([-1.0] * N_FEATURES)
                continue

            sum_valid_data = valid_data[:, 5].sum().item()
            rate = sum_valid_data / time_delta
            RA = np.deg2rad(valid_data[:, 1])
            mean_cos_RA = np.mean(np.cos(RA))
            mean_sin_RA = np.mean(np.sin(RA))
            R_RA = np.sqrt(mean_cos_RA ** 2 + mean_sin_RA ** 2)
            phase = valid_data[:, 7]
            mean_cos_phase = np.mean(np.cos(2 * np.pi * phase))
            mean_sin_phase = np.mean(np.sin(2 * np.pi * phase))
            R_phase = np.sqrt(mean_cos_phase ** 2 + mean_sin_phase ** 2)
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

            good_data_aggregated.append([sum_valid_data, time_delta, start_time, end_time,
                rate, mean_cos_RA, mean_sin_RA, R_RA, mean_cos_phase,
                mean_sin_phase, R_phase, mean_X_RE, mean_Y_RE, mean_Z_RE,
                std_X_RE, std_Y_RE, std_Z_RE, mean_R, std_R])

        return np.asarray(good_data_aggregated)

    def _extract_good_data_intervals(self, instruction_data, channel_index: int, device: str, option: str = "all_channels_working") -> List[Dict[str, Any]]:
        """
        Extract ENA acquisition time intervals from an instruction file according
        to channel availability and phase-coverage criteria.

        The method parses the instruction data row by row and identifies time
        intervals for which the full phase range (0–59) is covered. Depending on
        the selected ``option``, different channel-availability conditions are
        applied, and each returned interval may be marked for replacement.

        The output intervals are represented as dictionaries containing
        ``'start_time'``, ``'end_time'`` and a boolean ``'replace'`` flag.
        The ``replace`` flag indicates whether the channel was inactive during
        the interval and whether the corresponding data should be replaced
        (e.g. with ``-1``) in downstream aggregation steps.

        Supported modes of operation:

        - ``"all_channels_working"``:
          An interval is returned only if **all channels of the given device**
          (IBEX-Hi or IBEX-Lo) were operational. All returned intervals have
          ``replace = False``.

        - ``"replace_non_working_with_-1"``:
          All intervals with full phase coverage are returned. If the analyzed
          channel was inactive during the interval, it is marked with
          ``replace = True``; otherwise ``replace = False``.
          This mode preserves temporal continuity across channels while
          explicitly encoding missing measurements.

        - ``"single_channel_only"``:
          An interval is returned only if the analyzed channel was operational.
          All returned intervals have ``replace = False``.

        Parameters
        ----------
        instruction_data : np.ndarray
            Structured array loaded from the instruction file, containing time
            boundaries, phase coverage information, and channel availability flags.
        channel_index : int
            Index of the analyzed energy channel (0-based within the device).
        device : str
            Instrument identifier, either ``"Hi"`` (IBEX-Hi) or ``"Lo"`` (IBEX-Lo).
            Determines the number of channels and the interpretation of availability
            flags.
        option : str, optional
            Strategy controlling how channel availability is handled. Must be one
            of ``"all_channels_working"``, ``"replace_non_working_with_-1"``,
            or ``"single_channel_only"``. Default is ``"all_channels_working"``.

        Returns
        -------
        list of dict
            List of dictionaries describing valid acquisition intervals. Each
            dictionary contains:

            - ``'start_time'`` : float
              Start time of the interval (MET).
            - ``'end_time'`` : float
              End time of the interval (MET).
            - ``'replace'`` : bool
              Flag indicating whether data for this interval should be replaced
              (e.g. with ``-1``) due to channel inactivity.
        """
        double_observation = False
        phase_start_col = 3
        phase_end_col = 4
        start_bool_idx = 6
        if device == "Hi":
            n_channels = 6
            channel_index = channel_index -1
            print(channel_index)
        elif device == "Lo":
            n_channels = 8
            channel_index = channel_index-7
            print(channel_index)
        else:
            raise ValueError(f"Unknown device: {device}")
        end_bool_idx = start_bool_idx + n_channels - 1

        channel_bool_col = start_bool_idx + channel_index
        if channel_bool_col > end_bool_idx:
            channel_bool_col -= n_channels
        good_data_intervals = []

        for row in instruction_data:
            if not (row[phase_start_col] == 0 and row[phase_end_col] == 59):
                continue
            start_time = row[1]
            end_time = row[2]
            if option == "all_channels_working":
                flags_ok = all(row[f'channel_{i}'] == 1 for i in range(1, n_channels + 1))
                if device == "Hi":
                    double_observation = (row['double_obs'] == 2)
                    if flags_ok and double_observation:
                        good_data_intervals.append({"start_time": start_time, "end_time": end_time, "replace": False})
                else:
                    if flags_ok:
                        good_data_intervals.append({"start_time": start_time, "end_time": end_time, "replace": False})
            elif option == "replace_non_working_with_-1":
                replace = (row[channel_bool_col] == 0)
                good_data_intervals.append({"start_time": start_time, "end_time": end_time, "replace": replace})
            elif option == "single_channel_only":
                if row[channel_bool_col] == 1:
                    good_data_intervals.append({"start_time": start_time,"end_time": end_time,"replace": False})
            else:
                raise ValueError(f"Unknown option: {option}")
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
                    ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4'), ('double_obs', 'i4')]
        else:
            return [('orbit', 'i4'), ('start_time', 'f8'), ('end_time', 'f8'), ('phase_start', 'i4'),
                    ('phase_end', 'i4'), ('dataset', 'U2'), ('channel_1', 'i4'), ('channel_2', 'i4'),
                    ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4'),
                    ('channel_7', 'i4'), ('channel_8', 'i4')]

############################################################  deprecated functions  #################################################################################################
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