# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
import logging
import typing as tp
from abc import ABC, abstractmethod

import pandas as pd

from .utils import load_obj

logger = logging.getLogger(__name__)


class AnomalyDetector(ABC):
    """A class for detecting anomalies in a time series.

    Args:
        time_window (str): The time window for detecting anomalies, e.g. "1H".

    Returns:
        None

    Raises:
        NotImplementedError: If the detect method is not implemented by a subclass.

    """

    def __init__(
        self,
        time_window: str,
    ):
        self.time_window = time_window

    @abstractmethod
    def detect(self, data: pd.Series) -> pd.DataFrame:
        """Detect anomalies in a time series.

        Args:
            data: The time series to be checked for anomalies.

        Returns:
            A dataframe with at least the following columns:
             - "name": indicating the name of the tag/feature
             - "is_anomaly": indicating whether each data point is an anomaly
             - "anomaly_type": indicating the type of anomaly detected
             - "time_window": indicating the window in which the anomaly was detected
             - "comments": indicating any additional information about the anomaly
        """
        raise NotImplementedError(
            "The detect method must be implemented by a subclass.",
        )


class RangeDetector(AnomalyDetector):
    """A class for detecting anomalies based on given range.

    Attributes:
        time_window (str): The time window for detecting anomalies, e.g. "1H".
        threshold (float): The threshold for detecting anomalies.
        tag_range (dict): The range for detecting anomalies.

    Methods:
        detect(data): Detect anomalies in a time series based on a range.
    """

    def __init__(
        self,
        time_window: str,
        threshold: float,
        tag_range: tp.Dict[str, tp.Tuple[float, float]],
    ):
        super().__init__(time_window)
        self.threshold = threshold
        self.tag_range = tag_range

    def __repr__(self) -> str:
        return (
            f"RangeDetector(time_window={self.time_window}, "
            f"threshold={self.threshold}, tag_range={self.tag_range})"
        )

    def detect(self, data: pd.Series) -> pd.DataFrame:
        """Detect anomalies in a time series based on a range.

        Args:
            data: The time series with datetime index to be checked for anomalies.

        Returns:
            A dataframe with columns:
             - "name": indicating the name of the tag/feature
             - "is_anomaly": indicating whether each data point is an anomaly
             - "anomaly_type": indicating the type of anomaly detected
             - "time_window": indicating the window in which the anomaly was detected
             - "comments": indicating any additional information about the anomaly
             - "outlier_percentage": indicating the percentage of time the data is out
                of range
             - "lower_bound": indicating the lower bound of the anomaly window
             - "upper_bound": indicating the upper bound of the anomaly window
        """
        time_window = self.time_window
        threshold = self.threshold
        tag_name = data.name
        lower_bound = self.tag_range[tag_name][0]
        upper_bound = self.tag_range[tag_name][1]

        df_anomaly = pd.DataFrame(
            {
                "outlier_percentage": (
                    ~data.between(lower_bound, upper_bound)
                ).resample(time_window).mean(),
            },
        )

        df_anomaly["is_anomaly"] = df_anomaly["outlier_percentage"] > threshold
        df_anomaly["anomaly_type"] = "out of range"
        df_anomaly["comments"] = ""
        anomaly_comments = (
            f"{data.name} is out of range more than "
            f"{threshold = :.2%} of the time."
        )
        df_anomaly.loc[df_anomaly["is_anomaly"], "comments"] = anomaly_comments
        df_anomaly["name"] = tag_name
        df_anomaly["lower_bound"] = lower_bound
        df_anomaly["upper_bound"] = upper_bound
        df_anomaly["time_window"] = time_window

        return df_anomaly[
            [
                "name",
                "is_anomaly",
                "anomaly_type",
                "time_window",
                "comments",
                "outlier_percentage",
                "lower_bound",
                "upper_bound",
            ]
        ].reset_index()


class MissingValuesDetector(AnomalyDetector):
    """A class for detecting anomalies based on missing values.

    Attributes:
        time_window (str): The time window for detecting anomalies, e.g. "1H".
        threshold (float): The threshold for detecting anomalies.

    Methods:
        detect(data): Detect anomalies in a time series based on missing values.
    """

    def __init__(
        self,
        time_window: str,
        threshold: float,
    ):
        super().__init__(time_window)
        self.threshold = threshold

    def __repr__(self) -> str:
        return (
            f"MissingValuesDetector(time_window={self.time_window}, "
            f"threshold={self.threshold})"
        )

    def detect(self, data: pd.Series) -> pd.DataFrame:
        """Detect anomalies in a time series based on missing values.

        Args:
            data: The time series with datetime index to be checked for anomalies.

        Returns:
            A dataframe with columns:
             - "name": indicating the name of the tag/feature
             - "is_anomaly": indicating whether each data point is
              an anomaly
             - "anomaly_type": indicating the type of anomaly detected
             - "time_window": indicating the window in which the anomaly was detected
             - "comments": indicating any additional information about the anomaly
             - "missing_percentage": indicating the percentage of time the data is
              missing
        """
        time_window = self.time_window
        threshold = self.threshold

        df_anomaly = pd.DataFrame(
            {
                "missing_percentage": data.isna().resample(time_window).mean(),
            },
        )

        df_anomaly["is_anomaly"] = df_anomaly["missing_percentage"] > threshold
        df_anomaly["anomaly_type"] = "missing values"
        df_anomaly["comments"] = ""
        anomaly_comments = (
            f"{data.name} is missing more than "
            f"{threshold = :.2%} of the time."
        )
        df_anomaly.loc[df_anomaly["is_anomaly"], "comments"] = anomaly_comments
        df_anomaly["name"] = data.name
        df_anomaly["time_window"] = time_window

        return df_anomaly[
            [
                "name",
                "is_anomaly",
                "anomaly_type",
                "time_window",
                "comments",
                "missing_percentage",
            ]
        ].reset_index()
# TODO: Add more anomaly detectors such as variability detector, etc.


def detect_data_anomaly(
    data: pd.DataFrame,
    anomaly_detectors: tp.Dict[str, tp.List[AnomalyDetector]],
    timestamp_col: tp.Optional[str] = None,
) -> pd.Series:
    """Detect anomalies for given tags in the data.
    Args:
        data: The data to be checked for anomalies.
        anomaly_detectors: The anomaly detector to be used. It should be a dictionary
            where the keys are the names of the tags to be checked and the values are
            the lists of anomaly detectors to be used for each tag.
        timestamp_col: The name of the column with the timestamp if the data is not
            indexed by timestamp.
    Returns:
        A dataframe with at least the following columns:
         - "name": indicating the name of the tag/feature
         - "is_anomaly": indicating whether each data point is an anomaly
         - "time_window": indicating the window in which the anomaly was detected
         - "comments": indicating any additional information about the anomaly
    """
    df = _check_timestamp_col(data, timestamp_col)
    anomalies = pd.DataFrame()
    for tag, detectors in anomaly_detectors.items():
        if tag not in data:
            raise ValueError(f"Tag {tag} not found in the data.")
        for detector in detectors:
            tag_data = df[tag]
            tag_anomalies = detector.detect(tag_data)
            anomalies = pd.concat([anomalies, tag_anomalies], axis=0)

    return anomalies.reset_index(drop=True)


def _check_timestamp_col(
    data: pd.DataFrame,
    timestamp_col: tp.Optional[str] = None,
) -> pd.DataFrame:
    """Check if the timestamp column is in the data and set it as index if not.
    Args:
        data: The data to be checked for timestamp column.
        timestamp_col: The name of the column with the timestamp if the data is not
            indexed by timestamp.
    Returns:
        The data with timestamp column as index.
    """
    if timestamp_col is None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "The data should be indexed by timestamp or "
                "a timestamp column should be provided.",
            )
        return data.copy()
    elif timestamp_col not in data.columns:
        raise ValueError(f"Timestamp column {timestamp_col} not found in the data.")

    return data.set_index(timestamp_col).copy()


def create_detectors_dict(
    anomaly_parameters: dict[str, dict[str, str]],
    tags_to_monitor: tp.List[str],
) -> dict[str, list[AnomalyDetector]]:
    """
    Create a dictionary of anomaly detectors for each variable.
    The detectors are defined in the anomaly_parameters dictionary.
    Users can modify this function and the anomaly_parameters
    to allow different detectors for each variable.
    Args:
        anomaly_parameters: A dictionary that specifies what detectors to use and
            their parameters.
            >>> anomaly_parameters = {
            >>>     "preprocessing.MissingValuesDetector": {   # detector name
            >>>         "time_window": "3H",  # time window for detecting anomalies
            >>>         "threshold": 0.5,  # threshold for detecting anomalies
            >>>     },
            >>>     "preprocessing.RangeDetector": {  # detector name
            >>>         "time_window": "3H",  # time window for detecting anomalies
            >>>         "threshold": 0.5,  # threshold for detecting anomalies
            >>>         "tag_range": {  # range for detecting anomalies
            >>>             "iron_feed": [30, 70],  # range for iron_feed
            >>>             "silica_feed": [0, 35],  # range for silica_feed
            >>>             "ore_pulp_ph": [8.5, 12],  # range for ore_pulp_ph
            >>>         },
            >>>     },
            >>> }
        tags_to_monitor: A list of tags to monitor.
    Returns:
        A dictionary of anomaly detectors for each variable.
    """
    # instantiate anomaly detectors
    detectors = {}
    for detector_name, detector_parameters in anomaly_parameters.items():
        detectors[detector_name] = load_obj(detector_name)(**detector_parameters)

    # create a dictionary of anomaly detectors for each variable
    anomaly_detector_dict = {}
    for monitor_variable in tags_to_monitor:
        for detector in detectors.values():
            if monitor_variable not in anomaly_detector_dict:
                anomaly_detector_dict[monitor_variable] = [detector]
            else:
                anomaly_detector_dict[monitor_variable].append(detector)

    return anomaly_detector_dict
