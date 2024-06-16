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
Tests the data cleaning code
"""
import logging

import pandas as pd
import pytest

from preprocessing.utils import calculate_tag_range

logger = logging.getLogger(__name__)


@pytest.fixture()
def input_df_has_outliers_one_col() -> pd.DataFrame:
    """input dataframe to cleaning module. Outliers exist in column
    input"""

    data = [
        ["2017-04-02 00:00:00", -10],
        ["2017-04-02 01:00:00", 2],
        ["2017-04-02 01:00:00", 2],
        ["2017-04-02 02:00:00", 3],
        ["2017-04-02 03:00:00", 4],
        ["2017-04-02 04:00:00", 5],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_get_tag_range_min_max(
    input_df_has_outliers_one_col,
):
    """
    Test get_tag_range using min max method
    """
    got = calculate_tag_range(
        data=input_df_has_outliers_one_col,
        list_of_tags=["input"],
        method="min-max",
    )
    expected = {
        "input": (-10, 5),
    }
    assert got == expected


def test_get_tag_range_iqr(
    input_df_has_outliers_one_col,
):
    """
    Test get_tag_range using min max method
    """
    got = calculate_tag_range(
        data=input_df_has_outliers_one_col,
        list_of_tags=["input"],
        method="IQR",
    )
    expected = {
        "input": (-0.625, 5),
    }
    assert got == expected


def test_get_tag_range_three_sigma(
    input_df_has_outliers_one_col,
):
    """
    Test get_tag_range using min max method
    """
    got = calculate_tag_range(
        data=input_df_has_outliers_one_col,
        list_of_tags=["input"],
        method="3-sigma",
    )
    expected = {
        "input": (-10, 5),
    }
    assert got == expected
