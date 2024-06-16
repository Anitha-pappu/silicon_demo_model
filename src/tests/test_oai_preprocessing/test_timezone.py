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
Tests the timezone resolution code
"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from preprocessing.timezones import round_timestamps


@pytest.fixture()
def input_df_hour():
    """input dataframe to timezone convert"""

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:30:00", 2, 20, 200],
        ["2017-01-01 02:30:00", 3, 30, 300],
        ["2017-01-01 03:40:00", 4, 40, 400],
        ["2017-01-01 04:30:00", 5, 50, 500],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_min():
    """input dataframe to timezone convert"""

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:00:30", 2, 20, 200],
        ["2017-01-01 00:00:40", 3, 30, 300],
        ["2017-01-01 00:00:50", 4, 40, 400],
        ["2017-01-01 00:01:20", 5, 50, 500],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_sec():
    """input dataframe to timezone convert"""

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:00:05", 2, 20, 200],
        ["2017-01-01 00:00:10", 3, 30, 300],
        ["2017-01-01 00:00:25", 4, 40, 400],
        ["2017-01-01 00:00:30", 5, 50, 500],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_round_timestamps_hour(input_df_hour):

    result = round_timestamps(
        frequency="2h",
        data=input_df_hour,
        datetime_col="timestamp",
    )

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:00:00", 2, 20, 200],
        ["2017-01-01 02:00:00", 3, 30, 300],
        ["2017-01-01 04:00:00", 4, 40, 400],
        ["2017-01-01 04:00:00", 5, 50, 500],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert_frame_equal(result, expected)


def test_round_timestamps_min(input_df_min):

    result = round_timestamps(
        frequency="1t",
        data=input_df_min,
        datetime_col="timestamp",
    )

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:00:00", 2, 20, 200],
        ["2017-01-01 00:01:00", 3, 30, 300],
        ["2017-01-01 00:01:00", 4, 40, 400],
        ["2017-01-01 00:01:00", 5, 50, 500],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert_frame_equal(result, expected)


def test_round_timestamps_sec(input_df_sec):

    result = round_timestamps(
        frequency="30S",
        data=input_df_sec,
        datetime_col="timestamp",
    )

    data = [
        ["2017-01-01 00:00:00", 1, 10, 100],
        ["2017-01-01 00:00:00", 2, 20, 200],
        ["2017-01-01 00:00:00", 3, 30, 300],
        ["2017-01-01 00:00:30", 4, 40, 400],
        ["2017-01-01 00:00:30", 5, 50, 500],
    ]

    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert_frame_equal(result, expected)
