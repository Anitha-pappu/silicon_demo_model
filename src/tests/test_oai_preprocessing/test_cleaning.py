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

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import TypeAdapter

from preprocessing.cleaning import (
    deduplicate_pandas,
    enforce_schema,
    remove_null_columns,
    remove_outlier,
    replace_inf_values,
    unify_timestamp_col_name,
)
from preprocessing.tags_config import (
    TagMetaParameters,
    TagOutliersParameters,
    TagsConfig,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def input_df_all_null_columns():
    """input dataframe to cleaning module. One fully null column and
    one column with all but one None value to test that only the
    fully None column gets dropped"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, None, 1],
        ["2017-04-02 00:00:00", 1, 5, 3, None, None],
        ["2017-04-02 01:00:00", 2, 20, 200, None, None],
        ["2017-04-02 01:00:00", 2, 20, 200, None, None],
        ["2017-04-02 02:00:00", 3, 30, 300, None, None],
        ["2017-04-02 03:00:00", 4, 40, 400, None, None],
        ["2017-04-02 04:00:00", 5, 50, 500, None, None],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "input",
            "output",
            "control",
            "null_column",
            "almost_null",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_all_null_columns_base():
    """input dataframe to cleaning module. One fully null column and
    one column with all but one None value to test that only the
    fully None column gets dropped"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 1],
        ["2017-04-02 00:00:00", 1, 5, 3, 2],
        ["2017-04-02 01:00:00", 2, 20, 200, 3],
        ["2017-04-02 01:00:00", 2, 20, 200, 4],
        ["2017-04-02 02:00:00", 3, 30, 300, 5],
        ["2017-04-02 03:00:00", 4, 40, 400, 6],
        ["2017-04-02 04:00:00", 5, 50, 500, None],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "almost_null"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_duplicates():
    """input dataframe to cleaning module. One duplicated row and
    two duplicated column timestamp"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100],
        ["2017-04-02 00:00:00", 1, 5, 3],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_boolean():
    """input dataframe to cleaning module. columns that need to be converted to
    boolean exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, "yes", True],
        ["2017-04-02 00:00:00", 1, 5, 3, "yes", False],
        ["2017-04-02 01:00:00", 2, 20, 200, "yes", False],
        ["2017-04-02 01:00:00", 2, 20, 200, "no", False],
        ["2017-04-02 02:00:00", 3, 30, 300, "no", np.nan],
        ["2017-04-02 03:00:00", 4, 40, 400, "no", np.nan],
        ["2017-04-02 04:00:00", 5, 50, 500, "no", np.nan],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "bool_1", "bool_2"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_strange_boolean():
    """input dataframe to cleaning module. columns that need to be converted to
    boolean exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, "on"],
        ["2017-04-02 00:00:00", 1, 5, 3, "off"],
        ["2017-04-02 01:00:00", 2, 20, 200, "off"],
        ["2017-04-02 01:00:00", 2, 20, 200, "on"],
        ["2017-04-02 02:00:00", 3, 30, 300, "on"],
        ["2017-04-02 03:00:00", 4, 40, 400, "off"],
        ["2017-04-02 04:00:00", 5, 50, 500, np.nan],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "bool_1"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_boolean_map_str():
    """input dataframe to cleaning module. columns that need to be converted to
    boolean with explicit user specified mapping as a params"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, "Running"],
        ["2017-04-02 00:00:00", 1, 5, 3, "STOP"],
        ["2017-04-02 01:00:00", 2, 20, 200, "stop"],
        ["2017-04-02 01:00:00", 2, 20, 200, "ON"],
        ["2017-04-02 02:00:00", 3, 30, 300, "1"],
        ["2017-04-02 03:00:00", 4, 40, 400, "off"],
        ["2017-04-02 04:00:00", 5, 50, 500, np.nan],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "bool_2"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# TODO: no tests for this data
@pytest.fixture()
def input_df_has_float():
    """input dataframe to cleaning module. columns that need to be converted to
    integer exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 02:00:00", 3, 30, 300.0],
        ["2017-04-02 03:00:00", 4, 40, 400.0],
        ["2017-04-02 04:00:00", 5, 50, 500.0],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_int_with_missing_value():
    """input dataframe to cleaning module. columns that need to be converted to
    numeric exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 02:00:00", 3, 30, 300.0],
        ["2017-04-02 03:00:00", 4, 40, 400.0],
        ["2017-04-02 04:00:00", 5, None, 500.0],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# TODO: no tests for this data
@pytest.fixture()
def input_df_has_int_with_wrong_value():
    """input dataframe to cleaning module. columns that need to be converted to
    numeric exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 02:00:00", 3, 30, 300.0],
        ["2017-04-02 03:00:00", 4, 40, 400.0],
        ["2017-04-02 04:00:00", 5, None, "wrong"],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_str():
    """input dataframe to cleaning module. columns that need to be converted to
    str exist"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 111],
        ["2017-04-02 00:00:00", 1, 5, 3, 112],
        ["2017-04-02 01:00:00", 2, 20, 200, 113],
        ["2017-04-02 01:00:00", 2, 20, 200, 114],
        ["2017-04-02 02:00:00", 3, 30, 300, 115],
        ["2017-04-02 03:00:00", 4, 40, 400, 116],
        ["2017-04-02 04:00:00", 5, 50, 500, 117],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "category"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_datetime_str():
    """input dataframe to cleaning module. datetime column exists"""

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, "2017-04-02 04:07:00"],
        ["2017-04-02 00:00:00", 1, 20, 100, "2017-04-02 04:07:00"],
        ["2017-04-02 01:00:00", 2, 20, 200, "2017-04-02 04:07:00"],
        ["2017-04-02 01:00:00", 2, 20, 200, "2017-04-02 04:07:00"],
        ["2017-04-02 02:00:00", 3, 30, 300, "2017-04-02 04:07:00"],
        ["2017-04-02 03:00:00", 4, 40, 400, "2017-04-02 04:07:00"],
        ["2017-04-02 04:00:00", 5, 50, 500, "2017-04-02 04:07:00"],
    ]
    df = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "datetime_1"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_has_outliers():
    """input dataframe to cleaning module. Outliers exist in column
    input and control"""

    data = [
        ["2017-04-02 00:00:00", -10, 10, 501],
        ["2017-04-02 01:00:00", 2, 20, 9999],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def input_df_filter_period_has_inf():
    """
    Input data for
    """
    data = [
        ["2018-01-01 01:00:00", 10, None],
        ["2018-01-07 01:00:00", 10, 150],
        ["2018-01-14 01:00:00", 20, 250],
        ["2019-01-01 01:00:00", 30, 350],
        ["2021-01-01 01:00:00", 40, 300],
        ["2021-01-07 01:00:00", 50, 400],
        ["2021-02-14 01:00:00", np.inf, 50],
    ]
    df = pd.DataFrame(data, columns=["timestamp", "val_inf", "val_nan"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def tags_meta_config():
    raw_config = [
        {
            "raw_tag": "input rock quantity",
            "tag_name": "input",
            "data_type": "numeric",
            "data_source": "source a",
            "unit": "%",
            "min": 0,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "output rock quantity",
            "tag_name": "output",
            "data_type": "numeric",
            "data_source": "source a",
            "unit": "%",
            "min": None,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "mill b power",
            "tag_name": "control",
            "data_type": "numeric",
            "data_source": "source a",
            "unit": "%",
            "min": 200,
            "max": 500,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "on off mill a",
            "tag_name": "bool_1",
            "data_type": "boolean",
            "data_source": "source a",
            "unit": "%",
            "min": None,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "on off mill b",
            "tag_name": "bool_2",
            "data_type": "boolean",
            "data_source": "source a",
            "unit": "%",
            "min": None,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "some category",
            "tag_name": "category",
            "data_type": "categorical",
            "data_source": "source a",
            "unit": "%",
            "min": None,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
        {
            "raw_tag": "last checked",
            "tag_name": "timestamp_extra",
            "data_type": "datetime",
            "data_source": "source a",
            "unit": "%",
            "min": None,
            "max": None,
            "extract_freq": "5min",
            "tag_type": "input",
        },
    ]
    return TagsConfig(
        TypeAdapter(list[TagMetaParameters]).validate_python(raw_config),
        TagMetaParameters,
    )


@pytest.fixture
def tags_outliers_config():
    raw_config = [
        {
            "raw_tag": "input rock quantity",
            "tag_name": "input",
            "range_min": 0,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "output rock quantity",
            "tag_name": "output",
            "range_min": np.nan,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "mill b power",
            "tag_name": "control",
            "range_min": 200,
            "range_max": 500,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "on off mill a",
            "tag_name": "bool_1",
            "range_min": np.nan,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "on off mill b",
            "tag_name": "bool_2",
            "range_min": np.nan,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "some category",
            "tag_name": "category",
            "range_min": np.nan,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
        {
            "raw_tag": "last checked",
            "tag_name": "timestamp_extra",
            "range_min": np.nan,
            "range_max": np.nan,
            "special_values": "",
            "outlier_rules": "drop",
        },
    ]

    return TagsConfig(
        TypeAdapter(list[TagOutliersParameters]).validate_python(raw_config),
        TagOutliersParameters,
    )


def test_remove_null_columns(input_df_all_null_columns):
    """
    Test drop_null_columns function
    """

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 1],
        ["2017-04-02 00:00:00", 1, 5, 3, None],
        ["2017-04-02 01:00:00", 2, 20, 200, None],
        ["2017-04-02 01:00:00", 2, 20, 200, None],
        ["2017-04-02 02:00:00", 3, 30, 300, None],
        ["2017-04-02 03:00:00", 4, 40, 400, None],
        ["2017-04-02 04:00:00", 5, 50, 500, None],
    ]
    expected = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "almost_null"],
    )
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    got = remove_null_columns(input_df_all_null_columns)

    assert_frame_equal(got, expected)


def test_remove_null_columns_base(input_df_all_null_columns_base):
    """
    Test to ensure drop_null_columns does not drop columns that have values
    """

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 1],
        ["2017-04-02 00:00:00", 1, 5, 3, 2],
        ["2017-04-02 01:00:00", 2, 20, 200, 3],
        ["2017-04-02 01:00:00", 2, 20, 200, 4],
        ["2017-04-02 02:00:00", 3, 30, 300, 5],
        ["2017-04-02 03:00:00", 4, 40, 400, 6],
        ["2017-04-02 04:00:00", 5, 50, 500, None],
    ]
    expected = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "almost_null"],
    )
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    got = remove_null_columns(input_df_all_null_columns_base)

    assert_frame_equal(got, expected)


def test_unify_timestamp_col_name(input_df_has_datetime_str):
    """
    Test unify timestamp columns
    """

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, "2017-04-02 04:07:00"],
        ["2017-04-02 00:00:00", 1, 20, 100, "2017-04-02 04:07:00"],
        ["2017-04-02 01:00:00", 2, 20, 200, "2017-04-02 04:07:00"],
        ["2017-04-02 01:00:00", 2, 20, 200, "2017-04-02 04:07:00"],
        ["2017-04-02 02:00:00", 3, 30, 300, "2017-04-02 04:07:00"],
        ["2017-04-02 03:00:00", 4, 40, 400, "2017-04-02 04:07:00"],
        ["2017-04-02 04:00:00", 5, 50, 500, "2017-04-02 04:07:00"],
    ]
    expected = pd.DataFrame(
        data, columns=["unified_time_col", "input", "output", "control", "datetime_1"],
    )
    expected["unified_time_col"] = pd.to_datetime(expected["unified_time_col"])

    got = unify_timestamp_col_name(
        datetime_col="timestamp",
        data=input_df_has_datetime_str,
        unified_name="unified_time_col",
    )

    assert_frame_equal(got, expected, check_dtype=False, check_datetimelike_compat=True)


def test_unify_timestamp_col_name_fail(input_df_has_datetime_str):
    """
    Test unify timestamp columns fail
    """

    with pytest.raises(ValueError, match="column name 'timestamp' already exists"):
        unify_timestamp_col_name(
            datetime_col="datetime_1",
            data=input_df_has_datetime_str,
            unified_name="timestamp",
        )


def test_drop_duplicates_default(input_df_has_duplicates):
    """
    Test dropping duplicated rows
    """

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100],
        ["2017-04-02 00:00:00", 1, 5, 3],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])
    got = deduplicate_pandas(input_df_has_duplicates)

    assert_frame_equal(got, expected)


def test_drop_duplicates_customize(input_df_has_duplicates):
    """
    Test dropping rows with customized parameters
    """
    custom_kwargs = {"subset": "timestamp", "keep": "last"}

    data = [
        ["2017-04-02 00:00:00", 1, 5, 3],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])
    got = deduplicate_pandas(input_df_has_duplicates, **custom_kwargs)

    assert_frame_equal(got, expected)


def test_enforce_schema_bool(input_df_has_boolean, tags_meta_config):
    """
    Test convert dtypes to boolean
    """

    got = enforce_schema(data=input_df_has_boolean, meta_config=tags_meta_config)

    assert got["bool_1"].dtypes.name == "int64"


def test_enforce_schema_bool_with_missing_values(
    input_df_has_boolean,
    tags_meta_config,
):
    """
    Test convert dtypes to boolean causing error
    """

    got = enforce_schema(data=input_df_has_boolean, meta_config=tags_meta_config)

    assert got["bool_2"].dtypes.name == "float64"


def test_enforce_schema_bool_strange(input_df_has_strange_boolean, tags_meta_config):
    """
    Test convert dtypes to boolean
    """

    got = enforce_schema(
        data=input_df_has_strange_boolean,
        meta_config=tags_meta_config,
    )

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 1.0],
        ["2017-04-02 00:00:00", 1, 5, 3, 0.0],
        ["2017-04-02 01:00:00", 2, 20, 200, 0.0],
        ["2017-04-02 01:00:00", 2, 20, 200, 1.0],
        ["2017-04-02 02:00:00", 3, 30, 300, 1.0],
        ["2017-04-02 03:00:00", 4, 40, 400, 0.0],
        ["2017-04-02 04:00:00", 5, 50, 500, None],
    ]
    expected = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "bool_1"],
    )
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert got["bool_1"].dtypes.name == "float64"

    assert_frame_equal(got, expected, check_dtype=False)


def test_enforce_schema_map_str_to_bool(input_df_has_boolean_map_str, tags_meta_config):
    """
    Test convert to boolean with a user input mapping for strings
    """

    map_str_bool = {"running": True, "stop": False, "on": True, "1": True, "off": False}
    got = enforce_schema(
        data=input_df_has_boolean_map_str,
        meta_config=tags_meta_config,
        map_str_bool=map_str_bool,
    )

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100, 1],
        ["2017-04-02 00:00:00", 1, 5, 3, 0],
        ["2017-04-02 01:00:00", 2, 20, 200, 0],
        ["2017-04-02 01:00:00", 2, 20, 200, 1],
        ["2017-04-02 02:00:00", 3, 30, 300, 1],
        ["2017-04-02 03:00:00", 4, 40, 400, 0],
        ["2017-04-02 04:00:00", 5, 50, 500, np.nan],
    ]
    expected = pd.DataFrame(
        data, columns=["timestamp", "input", "output", "control", "bool_2"],
    )
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert_frame_equal(got, expected, check_dtype=False)


def test_enforce_schema_numeric(input_df_has_int_with_missing_value, tags_meta_config):
    """
    Test convert dtypes to numeric
    """

    got = enforce_schema(
        data=input_df_has_int_with_missing_value,
        meta_config=tags_meta_config,
    )

    data = [
        ["2017-04-02 00:00:00", 1, 10, 100.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 01:00:00", 2, 20, 200.0],
        ["2017-04-02 02:00:00", 3, 30, 300.0],
        ["2017-04-02 03:00:00", 4, 40, 400.0],
        ["2017-04-02 04:00:00", 5, None, 500.0],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert got["output"].dtypes.name == "float64"

    assert_frame_equal(got, expected, check_dtype=False)


def test_enforce_schema_categorical(input_df_has_str, tags_meta_config):
    """
    Test convert dtypes to categorical
    """

    got = enforce_schema(data=input_df_has_str, meta_config=tags_meta_config)

    assert got["category"].dtypes.name == "category"


def test_enforce_schema_datetime(input_df_has_datetime_str, tags_meta_config):
    """
    Test Not change timestamp, which was supposed to be taken care of in
    timezone enforcement
    """

    got = enforce_schema(
        data=input_df_has_datetime_str,
        meta_config=tags_meta_config,
    )

    assert got["datetime_1"].dtypes.name == "object"


def test_clip_outlier(input_df_has_outliers, tags_outliers_config):
    """
    Test remove outliers. Outliers are clipped to boundaries
    """
    got, summary = remove_outlier(
        data=input_df_has_outliers,
        outliers_config=tags_outliers_config,
        rule="clip",
    )

    data = [
        ["2017-04-02 00:00:00", 0, 10, 500],
        ["2017-04-02 01:00:00", 2, 20, 500],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "input", "output", "control"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])

    assert_frame_equal(got, expected, check_dtype=False)


def test_drop_outlier(input_df_has_outliers, tags_outliers_config):
    """
    Test remove outliers. Outliers are dropped and appropriate summary is returned
    """
    got_df, got_summary = remove_outlier(
        data=input_df_has_outliers,
        outliers_config=tags_outliers_config,
        rule="drop",
    )

    data = [
        ["2017-04-02 00:00:00", None, 10, None],
        ["2017-04-02 01:00:00", 2, 20, None],
        ["2017-04-02 01:00:00", 2, 20, 200],
        ["2017-04-02 02:00:00", 3, 30, 300],
        ["2017-04-02 03:00:00", 4, 40, 400],
        ["2017-04-02 04:00:00", 5, 50, 500],
    ]

    expected_df = pd.DataFrame(
        data,
        columns=["timestamp", "input", "output", "control"],
    )
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])
    expected_summary = pd.DataFrame(
        {
            "tag_name": ["input", "control"],
            "outlier_percentage": [16.67, 33.33],
        },
    )

    assert_frame_equal(got_df, expected_df, check_dtype=False)
    assert_frame_equal(
        got_summary.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
        check_dtype=False,
    )


def test_drop_outlier_fail(input_df_has_outliers, tags_outliers_config):
    """
    Test drop outlier causing error
    """
    with pytest.raises(ValueError, match="Invalid outlier removal rule `drooop`"):
        remove_outlier(
            data=input_df_has_outliers,
            outliers_config=tags_outliers_config,
            rule="drooop",
        )


def test_replace_inf_values(input_df_filter_period_has_inf):
    """
    Test replace_inf_values function
    """

    got = replace_inf_values(input_df_filter_period_has_inf)

    data = [
        ["2018-01-01 01:00:00", 10, np.nan],
        ["2018-01-07 01:00:00", 10, 150],
        ["2018-01-14 01:00:00", 20, 250],
        ["2019-01-01 01:00:00", 30, 350],
        ["2021-01-01 01:00:00", 40, 300],
        ["2021-01-07 01:00:00", 50, 400],
        ["2021-02-14 01:00:00", np.nan, 50],
    ]
    expected = pd.DataFrame(data, columns=["timestamp", "val_inf", "val_nan"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])
    assert_frame_equal(got, expected, check_dtype=False)
