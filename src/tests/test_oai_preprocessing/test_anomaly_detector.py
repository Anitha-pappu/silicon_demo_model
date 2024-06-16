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
"""
Test the anomaly detector code
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from preprocessing import (
    MissingValuesDetector,
    RangeDetector,
    anomaly_detector,
    create_detectors_dict,
    detect_data_anomaly,
)


class TestRangeDetector(object):
    @pytest.fixture(scope="session")
    def input_df_outrange(self) -> pd.Series:
        dates = [
            "2023-02-16 12:00:00",
            "2023-02-16 12:01:00",
            "2023-02-16 12:02:00",
            "2023-02-16 12:03:00",
            "2023-02-16 12:04:00",
            "2023-02-16 12:05:00",
            "2023-02-16 12:06:00",
            "2023-02-16 12:07:00",
            "2023-02-16 12:08:00",
            "2023-02-16 12:09:00",
            "2023-02-16 12:10:00",
            "2023-02-16 12:11:00",
            "2023-02-16 12:12:00",
            "2023-02-16 12:13:00",
            "2023-02-16 12:14:00",
        ]

        value_col = [
            3,
            1,
            2,
            3,
            1,
            1,
            2,
            3,
            4,
            2,
            1,
            10,
            15,
            1,
            30,
        ]
        ts = pd.Series(data=value_col, index=pd.to_datetime(dates), name="value")
        return ts

    @pytest.fixture(scope="session")
    def range_detector(self) -> anomaly_detector.RangeDetector:
        time_window = "5T"
        threshold = 0.5
        tag_range = {"value": (1, 4)}
        return anomaly_detector.RangeDetector(time_window, threshold, tag_range)

    def test_detect(self, range_detector, input_df_outrange):
        expected = pd.DataFrame(
            {
                "index": pd.to_datetime([
                    "2023-02-16 12:00:00",
                    "2023-02-16 12:05:00",
                    "2023-02-16 12:10:00",
                ]),
                "name": ["value", "value", "value"],
                "is_anomaly": [
                    False,
                    False,
                    True,
                ],
                "anomaly_type": [
                    "out of range",
                    "out of range",
                    "out of range",
                ],
                "time_window": ["5T", "5T", "5T"],
                "comments": [
                    "",
                    "",
                    "value is out of range more than"
                    " threshold = 50.00% of the time.",
                ],
                "outlier_percentage": [
                    0.0,
                    0.0,
                    0.6,
                ],
                "lower_bound": [1, 1, 1],
                "upper_bound": [4, 4, 4],
            },
        )
        result = range_detector.detect(input_df_outrange)
        assert_frame_equal(result, expected)


class TestMissingValuesDetector(object):
    @pytest.fixture(scope="session")
    def input_df_missing(self) -> pd.Series:
        dates = [
            "2023-02-16 12:00:00",
            "2023-02-16 12:01:00",
            "2023-02-16 12:02:00",
            "2023-02-16 12:03:00",
            "2023-02-16 12:04:00",
            "2023-02-16 12:05:00",
            "2023-02-16 12:06:00",
            "2023-02-16 12:07:00",
            "2023-02-16 12:08:00",
            "2023-02-16 12:09:00",
            "2023-02-16 12:10:00",
            "2023-02-16 12:11:00",
            "2023-02-16 12:12:00",
            "2023-02-16 12:13:00",
            "2023-02-16 12:14:00",
        ]

        value_col = [
            3,
            1,
            2,
            3,
            1,
            1,
            2,
            3,
            4,
            2,
            1,
            np.nan,
            np.nan,
            1,
            np.nan,
        ]
        ts = pd.Series(data=value_col, index=pd.to_datetime(dates), name="value")
        return ts

    @pytest.fixture(scope="session")
    def missing_values_detector(self) -> anomaly_detector.MissingValuesDetector:
        time_window = "5T"
        threshold = 0.5
        return anomaly_detector.MissingValuesDetector(time_window, threshold)

    def test_detect(self, missing_values_detector, input_df_missing):
        expected = pd.DataFrame(
            {
                "index": pd.to_datetime([
                    "2023-02-16 12:00:00",
                    "2023-02-16 12:05:00",
                    "2023-02-16 12:10:00",
                ]),
                "name": "value",
                "is_anomaly": [
                    False,
                    False,
                    True,
                ],
                "anomaly_type": [
                    "missing values",
                    "missing values",
                    "missing values",
                ],
                "time_window": "5T",
                "comments": [
                    "",
                    "",
                    "value is missing more than"
                    " threshold = 50.00% of the time.",
                ],
                "missing_percentage": [
                    0.0,
                    0.0,
                    0.6,
                ],
            },
        )
        result = missing_values_detector.detect(input_df_missing)
        assert_frame_equal(result, expected)


class TestAnomalyDetection(object):
    @pytest.fixture(scope="session")
    def input_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": pd.to_datetime([
                    "2023-02-16 12:00:00",
                    "2023-02-16 12:01:00",
                    "2023-02-16 12:02:00",
                    "2023-02-16 12:03:00",
                    "2023-02-16 12:04:00",
                ]),
                "var1": [3, np.nan, np.nan, np.nan, 5],
                "var2": [28, 40, 30, 20, 10],
            },
        ).set_index("index")

    @pytest.fixture(scope="session")
    def anomaly_detectors(self) -> dict:
        time_window = "5T"
        threshold = 0.5
        range_detector = anomaly_detector.RangeDetector(
            time_window,
            threshold,
            {"var1": (2, 8), "var2": (4, 40)},
        )
        missing_values_detector = anomaly_detector.MissingValuesDetector(
            time_window,
            threshold,
        )
        return {
            "var1": [range_detector, missing_values_detector],
            "var2": [range_detector],
        }

    def test_detect_anomalies(self, input_data, anomaly_detectors):
        expected = pd.DataFrame(
            {
                "index": pd.to_datetime([
                    "2023-02-16 12:00:00",
                    "2023-02-16 12:00:00",
                    "2023-02-16 12:00:00",
                ]),
                "name": [
                    "var1", "var1", "var2",
                ],
                "is_anomaly": [
                    True,
                    True,
                    False,
                ],
                "anomaly_type": [
                    "out of range",
                    "missing values",
                    "out of range",
                ],
                "time_window": ["5T", "5T", "5T"],
                "comments": [
                    "var1 is out of range more than"
                    " threshold = 50.00% of the time.",
                    "var1 is missing more than"
                    " threshold = 50.00% of the time.",
                    "",
                ],
                "outlier_percentage": [
                    0.6,
                    np.nan,
                    0,
                ],
                "lower_bound": [2, np.nan, 4],
                "upper_bound": [8, np.nan, 40],
                "missing_percentage": [
                    np.nan,
                    0.6,
                    np.nan,
                ],
            },
        )
        # Test anomaly detection
        result = detect_data_anomaly(input_data, anomaly_detectors)
        assert_frame_equal(result, expected)


class TestCreateDetectorsDict(object):
    @pytest.fixture(scope="session")
    def config(self) -> dict:
        return {
            "preprocessing.RangeDetector": {
                "time_window": "5T",
                "threshold": 0.5,
                "tag_range": {"var1": (2, 8), "var2": (4, 40)},
            },
            "preprocessing.MissingValuesDetector": {
                "time_window": "5T",
                "threshold": 0.5,
            },
        }

    def test_create_detectors_dict(self, config):
        expected = {
            "var1": [
                RangeDetector(
                    "5T",
                    0.5,
                    {"var1": (2, 8), "var2": (4, 40)},
                ),
                MissingValuesDetector("5T", 0.5),
            ],
            "var2": [
                RangeDetector(
                    "5T",
                    0.5,
                    {"var1": (2, 8), "var2": (4, 40)},
                ),
                MissingValuesDetector("5T", 0.5),
            ],
        }
        result = create_detectors_dict(config, ["var1", "var2"])
        assert print(result) == print(expected)
