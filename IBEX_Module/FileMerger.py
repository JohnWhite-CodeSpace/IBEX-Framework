import numpy as np
import yaml
import os

__author__ = "Jan Biały"

class FileMerger:
    """
    FileMerger
    ==========

    Class responsible for merging preprocessed IBEX data files into unified,
    channel-specific output files.

    The class operates on data batches generated in earlier stages of the
    analysis pipeline and combines them into single merged datasets. This
    approach enables processing large data volumes while limiting system
    resource usage, in particular RAM memory consumption.

    ------
    """
    def __init__(self, config: str):
        """
        Initialize the FileMerger object using a YAML configuration file.

        The configuration file specifies input file templates and the output
        directory used for merging channel-specific datasets.

        Parameters
        ----------
        config : str
            Path to the YAML configuration file.

        Raises
        ------
        Exception
            If the configuration file cannot be loaded or parsed.
        """
        try:
            with open(config, 'r') as cfg:
                self.cfg = yaml.load(cfg, Loader=yaml.FullLoader)
        except Exception as ex:
            self.WriteLog(-1, str(repr(ex)))

    def merge_files(self, channel_num:int = 15)-> dict | None:
        """
        Merge preprocessed IBEX data files into channel-specific merged datasets.

        For each energy channel, the method loads partial data files defined in the
        configuration, concatenates them vertically, and saves the merged result
        to the output directory.

        Parameters
        ----------
        channel_num : int, optional
            Upper bound of channel indices to process. Channels are iterated from
            1 to ``channel_num - 1``. Default is 15 (corresponding to 14 channels).

        Returns
        -------
        dict or None
            Dictionary containing ``return_code`` and ``error_message`` in case
            of an error. Returns ``None`` if all channels are merged successfully.
        """
        output_dir = self.cfg['merged_data_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        for i in range(1, channel_num):
            merged_data = []
            for folder_path, in self.cfg["folders"].items():
                file_path = folder_path.format(i)
                data = np.loadtxt(file_path)
                merged_data.append(data)
            if merged_data:
                try:
                    merged_array = np.vstack(merged_data)
                    output_path = os.path.join(output_dir, f"channel_{i}_merged.txt")
                    np.savetxt(output_path, merged_array, fmt="%.6f")
                    self.write_message(f"Merged data saved to {output_path}")
                except Exception as ex:
                    self.write_log(-1, str(repr(ex)))
                    return {"return_code": -1, 'error_message': repr(ex)}
            else:
                self.write_log(-2, f"Channel {i} has no data")
                return {"return_code": -2, 'error_message': f"Channel {i} has no data"}

    def write_log(self, return_code: int, error_message: str)-> None:
        """
        Log an error message and return code to the execution terminal.

        Parameters
        ----------
        return_code : int
            Numeric code describing the error type.
        error_message : str
            Description of the encountered error.

        Returns
        -------
        None
        """
        response_dict = {"return_code": return_code, 'error_message': error_message}
        print(f" FileMergerLog: {response_dict}")

    def write_message(self, message: str)-> None:
        """
        Log a status or informational message to the execution terminal.

        Parameters
        ----------
        message : str
            Informational message to be logged.

        Returns
        -------
        None
        """
        print(f"FileMergerLog: {message}")