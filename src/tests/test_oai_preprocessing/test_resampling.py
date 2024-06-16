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
Tests the export nodes
"""
from io import StringIO
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import TypeAdapter

from preprocessing.resampling import _get_valid_agg_method, resample_data
from preprocessing.tags_config import TagResampleParameters, TagsConfig


def make_df(csv_str):
    df = pd.read_csv(StringIO(dedent(csv_str)))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_data():
    """Mock input dataframe"""

    df = make_df(
        """
        timestamp,input,output,control
        2017-04-02 00:00:00,1,5,100
        2017-04-02 00:15:00,2,20,100
        2017-04-02 00:30:00,2,20,100
        2017-04-02 00:45:00,3,30,200
        2017-04-02 01:00:00,4,40,200
        2017-04-02 01:15:00,5,50,300
        2017-04-02 01:30:00,5,60,200
        2017-04-02 01:45:00,1.5,15,300
        """,
    )

    return df


@pytest.fixture
def resample_config():
    raw_config = [
        {
            "tag_name": "input",
            "raw_tag": "input",
            "resample_method": "mean",
            "resample_freq": "1H",
            "resample_offset": "0min",
        },
        {
            "tag_name": "output",
            "raw_tag": "output",
            "resample_method": "sum",
            "resample_freq": "1H",
            "resample_offset": "0min",
        },
        {
            "tag_name": "control",
            "raw_tag": "control",
            "resample_method": "mean",
            "resample_freq": "1H",
            "resample_offset": "0min",
        },
    ]

    return TagsConfig(
        TypeAdapter(list[TagResampleParameters]).validate_python(raw_config),
        TagResampleParameters,
    )


@pytest.fixture
def valid_tag_df_missing_agg():
    """A complete dataframe which can be converted into a TagDict"""
    df = pd.DataFrame(
        {
            "agg_method": ["mean", "sum", np.nan],
            "tag": ["input", "output", "control"],
            "data_type": ["numeric", "numeric", "numeric"],
            "range_min": [0, None, 200],
            "range_max": [None, None, 500],
            "name": ["input rock quantity", "output rock quantity", "mill b power"],
        },
    )
    return df


@pytest.fixture
def valid_tag_df_invalid_agg():
    """A complete dataframe which can be converted into a TagDict"""
    df = pd.DataFrame(
        {
            "agg_method": ["mean", "sum", "hello_world"],
            "tag": ["input", "output", "control"],
            "data_type": ["numeric", "numeric", "numeric"],
            "range_min": [0, None, 200],
            "range_max": [None, None, 500],
            "name": ["input rock quantity", "output rock quantity", "mill b power"],
        },
    )
    return df


@pytest.fixture()
def params_res():
    """Create simplified version of input parameters"""
    params = {
        "resample_kwargs": {
            "rule": "1H",  # Major time grid frequency to resample your dataset to
        },
        "errors": "raise",
        "default_method": "mean",
    }
    return params


def test_resample_data_working(params_res, input_data, resample_config):
    """Test resampling of dataset using tag_dict agg_method"""

    expected = make_df(
        """
        timestamp,input,output,control
        2017-04-02 00:00:00,2,75,125
        2017-04-02 01:00:00,3.875,165,250
        """,
    )

    got = resample_data(
        data=input_data,
        resample_config=resample_config,
        timestamp_col="timestamp",
        errors=params_res["errors"],
        default_method=params_res["default_method"],
    )

    assert_frame_equal(got[expected.columns], expected, check_dtype=False)


def test_resolve_to_default_agg_method(resample_config):
    """Test resolving to default aggregation method"""

    expected = "mean"
    tag_missing = "control"

    got = _get_valid_agg_method(
        tag_missing, resample_config, errors="coerce", default_method=expected,
    )

    assert expected == got


def test_resolve_to_default_agg_method_invalid_tag(resample_config):
    """Test resolving to default aggregation method for invalid agg_method"""

    expected = "mean"
    tag_invalid = "control"

    got = _get_valid_agg_method(
        tag_invalid, resample_config, errors="coerce", default_method=expected,
    )

    assert expected == got


def test_catch_missing_agg_method(resample_config):
    """Test failing with missing agg_method in data dictionary"""

    tag_missing = "control_missing"
    with pytest.raises(KeyError, match="'control_missing'"):
        _get_valid_agg_method(
            tag_missing,
            resample_config,
            errors="raise",
        )
