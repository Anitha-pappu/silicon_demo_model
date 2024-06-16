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
Tests the on/off logic code
"""
import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import TypeAdapter

from preprocessing.on_off_logic import set_off_equipment_to_zero
from preprocessing.tags_config import (
    TagMetaParameters,
    TagOnOffDependencyParameters,
    TagsConfig,
)

# @pytest.fixture()
# def sensor_df():
#     """input dataframe to test on off logic"""
#
#     data = [
#         ["2017-01-01 00:00:00", 0, 80, 200, 1, 0],
#         ["2017-01-01 01:00:00", 0, 70, 200, 2, 0],
#         ["2017-01-01 02:00:00", 300, 60, 210, 4, 1],
#         ["2017-01-01 03:00:00", 310, 50, 220, 8, 1],
#         ["2017-01-01 04:00:00", 320, 50, 100, 16, 1],
#         ["2017-01-01 05:00:00", 320, 50, 99, 32, 1],
#         ["2017-01-01 06:00:00", 0, 30, 0, 64, 0],
#         ["2017-01-01 07:00:00", 0, 20, 0, 128, 0],
#         ["2017-01-01 08:00:00", 200, 10, 0, 256, 1],
#     ]
#     df = pd.DataFrame(
#         data,
#         columns=[
#             "timestamp",
#             "mill_a_power",
#             "mill_a_load",
#             "mill_b_power",
#             "mill_b_load",
#             "on_off_mill_a",
#         ],
#     )
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     return df


# @pytest.fixture()
# def sensor_df_minute_level():
#     """input dataframe at minute level to test on off logic"""
#
#     data = [
#         ["2017-01-01 00:01:00", 0, 80, 200, 1, 0],
#         ["2017-01-01 00:02:00", 0, 70, 200, 2, 0],
#         ["2017-01-01 00:03:00", 300, 60, 210, 4, 1],
#         ["2017-01-01 00:04:00", 310, 50, 220, 8, 1],
#         ["2017-01-01 00:05:00", 320, 50, 100, 16, 1],
#         ["2017-01-01 00:06:00", 320, 50, 99, 32, 1],
#         ["2017-01-01 00:07:00", 0, 30, 0, 64, 0],
#         ["2017-01-01 00:08:00", 0, 20, 0, 128, 0],
#         ["2017-01-01 00:09:00", 0.01, 10, 500, 256, 0],
#         ["2017-01-01 00:10:00", 0.1, 10, 500, 256, 0],
#         ["2017-01-01 00:11:00", 2.5, 10, 21, 256, 0],
#         ["2017-01-01 00:12:00", 10, 10, 20, 256, 0],
#         ["2017-01-01 00:13:00", 11.5, 10, 19, 256, 0],
#         ["2017-01-01 00:14:00", 12.5, 10, 18, 256, 0],
#         ["2017-01-01 00:15:00", 19, 10, 5, 256, 0],
#         ["2017-01-01 00:16:00", 50, 50, 2, 256, 1],
#         ["2017-01-01 00:17:00", 60, 10, 1, 256, 1],
#         ["2017-01-01 00:18:00", 70, 80, 0.5, 256, 1],
#         ["2017-01-01 00:19:00", 80, 10, 0.3, 256, 1],
#         ["2017-01-01 00:20:00", 90, 100, 0.2, 256, 1],
#         ["2017-01-01 00:21:00", 100, 100, 0.1, 256, 1],
#         ["2017-01-01 00:22:00", 110, 200, 0, 256, 1],
#         ["2017-01-01 00:23:00", 110, 300, 200, 256, 1],
#         ["2017-01-01 00:24:00", 110, 300, 200, 256, 1],
#         ["2017-01-01 00:25:00", 110, 300, 200, 256, 1],
#         ["2017-01-01 00:26:00", 180, 300, 6, 256, 1],
#         ["2017-01-01 00:27:00", 190, 300, 500, 256, 1],
#         ["2017-01-01 00:28:00", 200, 350, 500, 256, 1],
#         ["2017-01-01 00:29:00", 210, 360, 500, 256, 1],
#         ["2017-01-01 00:30:00", 250, 380, 500, 256, 1],
#     ]
#     df = pd.DataFrame(
#         data,
#         columns=[
#             "timestamp",
#             "mill_a_power",
#             "mill_a_load",
#             "mill_b_power",
#             "mill_b_load",
#             "on_off_mill_a",
#         ],
#     )
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     return df


@pytest.fixture()
def sensor_with_on_off_df():
    """
    output dataframe with on_off tag added for Mill B
    """

    data = [
        ["2017-01-01 00:00:00", 0, 80, 200, 1, 0, 0],
        ["2017-01-01 01:00:00", 0, 70, 200, 2, 0, 0],
        ["2017-01-01 02:00:00", 300, 60, 210, 4, 1, 1],
        ["2017-01-01 03:00:00", 310, 50, 220, 8, 1, 1],
        ["2017-01-01 04:00:00", 320, 50, 100, 16, 1, 1],
        ["2017-01-01 05:00:00", 320, 50, 99, 32, 1, 0],
        ["2017-01-01 06:00:00", 0, 30, 0, 64, 0, 0],
        ["2017-01-01 07:00:00", 0, 20, 0, 128, 0, 0],
        ["2017-01-01 08:00:00", 200, 10, 0, 256, 1, 0],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "mill_a_power",
            "mill_a_load",
            "mill_b_power",
            "mill_b_load",
            "on_off_mill_a",
            "on_off_mill_b",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture()
def sensor_with_on_off_minute_level_df():
    """input dataframe at minute level to test on off logic"""

    data = [
        ["2017-01-01 00:01:00", 0, 80, 200, 1, 0, 0],
        ["2017-01-01 00:02:00", 0, 70, 200, 2, 0, 0],
        ["2017-01-01 00:03:00", 300, 60, 210, 4, 1, None],
        ["2017-01-01 00:04:00", 310, 50, 220, 8, 1, None],
        ["2017-01-01 00:05:00", 320, 50, 100, 16, 1, None],
        ["2017-01-01 00:06:00", 320, 50, 99, 32, 1, None],
        ["2017-01-01 00:07:00", 0, 30, 0, 64, 0, 0],
        ["2017-01-01 00:08:00", 0, 20, 0, 128, 0, 0],
        ["2017-01-01 00:09:00", 0.01, 10, 500, 256, 0, 0],
        ["2017-01-01 00:10:00", 0.1, 10, 500, 256, 0, 0],
        ["2017-01-01 00:11:00", 2.5, 10, 21, 256, 0, 0],
        ["2017-01-01 00:12:00", 10, 10, 20, 256, 0, 0],
        ["2017-01-01 00:13:00", 11.5, 10, 19, 256, 0, 0],
        ["2017-01-01 00:14:00", 12.5, 10, 18, 256, 0, 0],
        ["2017-01-01 00:15:00", 19, 10, 5, 256, 0, 0],
        ["2017-01-01 00:16:00", 50, 50, 2, 256, 1, 0],
        ["2017-01-01 00:17:00", 60, 10, 1, 256, 1, 0],
        ["2017-01-01 00:18:00", 70, 80, 0.5, 256, 1, 0],
        ["2017-01-01 00:19:00", 80, 10, 0.3, 256, 1, 0],
        ["2017-01-01 00:20:00", 90, 100, 0.2, 256, 1, 0],
        ["2017-01-01 00:21:00", 100, 100, 0.1, 256, 1, 0],
        ["2017-01-01 00:22:00", 110, 200, 0, 256, 1, 0],
        ["2017-01-01 00:23:00", 110, 300, 200, 256, 1, 0],
        ["2017-01-01 00:24:00", 110, 300, 200, 256, 1, 1],
        ["2017-01-01 00:25:00", 110, 300, 200, 256, 1, 1],
        ["2017-01-01 00:26:00", 180, 300, 6, 256, 1, 1],
        ["2017-01-01 00:27:00", 190, 300, 500, 256, 1, 1],
        ["2017-01-01 00:28:00", 200, 350, 500, 256, 1, 1],
        ["2017-01-01 00:29:00", 210, 360, 500, 256, 1, 1],
        ["2017-01-01 00:30:00", 250, 380, 500, 256, 1, 1],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "mill_a_power",
            "mill_a_load",
            "mill_b_power",
            "mill_b_load",
            "on_off_mill_a",
            "on_off_mill_b",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# @pytest.fixture
# def valid_tag_df():
#     """A complete dataframe which can be converted into a TagDict"""
#     df = pd.DataFrame(
#         {
#             "tag": [
#                 "mill_a_power",
#                 "mill_a_load",
#                 "mill_b_power",
#                 "mill_b_load",
#                 "on_off_mill_a",
#             ],
#             "name": [
#                 "mill a power",
#                 "mill a load",
#                 "mill b power",
#                 "mill b load",
#                 "on off mill a",
#             ],
#             "area": ["mill", "mill", "mill", "mill", "mill"],
#             "sub_area": ["mill a", "mill a", "mill b", "mill b", "mill a"],
#             "tag_type": ["control", "state", "control", "state", "on_off"],
#             "data_type": ["numeric", "numeric", "numeric", "numeric", "boolean"],
#             "unit": ["Power Units", "Rock Units", "Power Units", "Rock Units", None],
#             "range_min": [0, 0, 0, 0, None],
#             "range_max": [500, 1000, 500, 1000, None],
#             "op_min": [70, None, 90, None, None],
#             "op_max": [160, None, 310, None, None],
#             "on_off_dependencies": [
#                 "on_off_mill_a",
#                 "on_off_mill_a",
#                 "on_off_mill_a",
#                 None,
#                 None,
#             ],
#             "notes": [None, None, None, None, "a note"],
#             "derived": [None, None, None, None, None],
#         },
#     )
#
#     return df


@pytest.fixture
def tags_meta_config():
    raw_config = [
        {
            "raw_tag": "mill a power",
            "tag_name": "mill_a_power",
            "data_type": "numeric",
            "unit": "Power Units",
            "min": 0,
            "max": 500,
            "extract_freq": "5min",
            "tag_type": "control",
        },
        {
            "raw_tag": "mill a load",
            "tag_name": "mill_a_load",
            "data_type": "numeric",
            "unit": "Rock Units",
            "min": 0,
            "max": 1000,
            "extract_freq": "5min",
            "tag_type": "state",
        },
        {
            "raw_tag": "mill b power",
            "tag_name": "mill_b_power",
            "data_type": "numeric",
            "unit": "Power Units",
            "min": 0,
            "max": 500,
            "extract_freq": "5min",
            "tag_type": "control",
        },
        {
            "raw_tag": "mill b load",
            "tag_name": "mill_b_load",
            "data_type": "numeric",
            "unit": "Rock Units",
            "min": 0,
            "max": 1000,
            "extract_freq": "5min",
            "tag_type": "state",
        },
        {
            "raw_tag": "on off mill a",
            "tag_name": "on_off_mill_a",
            "data_type": "boolean",
            "unit": None,
            "min": None,
            "max": None,
            "extract_freq": None,
            "tag_type": "on_off",
        },
        {
            "raw_tag": "on off mill b",
            "tag_name": "on_off_mill_b",
            "data_type": "boolean",
            "unit": None,
            "min": None,
            "max": None,
            "extract_freq": None,
            "tag_type": "on_off",
        },
    ]

    return TagsConfig(
        TypeAdapter(list[TagMetaParameters]).validate_python(raw_config),
        TagMetaParameters,
    )


@pytest.fixture
def tags_on_off_config():
    raw_config = [
        {
            "tag_name": "mill_a_power",
            "on_off_dependencies": None,
        },
        {
            "tag_name": "mill_a_load",
            "on_off_dependencies": [
                "on_off_mill_a",
            ],
        },
        {
            "tag_name": "mill_b_power",
            "on_off_dependencies": [
                "on_off_mill_a",
                "on_off_mill_b",
            ],
        },
        {
            "tag_name": "mill_b_load",
            "on_off_dependencies": [
                "on_off_mill_b",
            ],
        },
        {
            "tag_name": "on_off_mill_a",
            "on_off_dependencies": None,
        },
        {
            "tag_name": "on_off_mill_b",
            "on_off_dependencies": None,
        },
    ]

    return TagsConfig(
        TypeAdapter(list[TagOnOffDependencyParameters]).validate_python(raw_config),
        TagOnOffDependencyParameters,
    )


# @pytest.fixture
# def valid_tag_df_with_on_off(valid_tag_df):
#     """
#     Valid tag DF + additional row for synthetic on/off tag + adjusted
#     mill_b_power to utilise this on/off tag
#     """
#     adjusted_tag_df = valid_tag_df.append(
#         {
#             "tag": "on_off_mill_b",
#             "name": "on off mill b",
#             "area": "mill",
#             "sub_area": "mill b",
#             "tag_type": "on_off",
#             "data_type": "boolean",
#             "derived": True,
#         },
#         ignore_index=True,
#     )
#
#     adjusted_tag_df.loc[2, "on_off_dependencies"] = "on_off_mill_a, on_off_mill_b"
#     adjusted_tag_df.loc[3, "on_off_dependencies"] = "on_off_mill_b"
#     return adjusted_tag_df


@pytest.fixture
def mark_tags_off_output_df():
    """
    Output df for marking tags as off
    :return:
    """

    data = [
        ["2017-01-01 00:00:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 01:00:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 02:00:00", 300, 60, 210, 4, 1, 1],
        ["2017-01-01 03:00:00", 310, 50, 220, 8, 1, 1],
        ["2017-01-01 04:00:00", 320, 50, 100, 16, 1, 1],
        ["2017-01-01 05:00:00", 320, 50, 0, 0, 1, 0],
        ["2017-01-01 06:00:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 07:00:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 08:00:00", 200, 10, 0, 0, 1, 0],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "mill_a_power",
            "mill_a_load",
            "mill_b_power",
            "mill_b_load",
            "on_off_mill_a",
            "on_off_mill_b",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def mark_tags_off_output_df_minute_level():
    """
    Output df for marking tags as off, at the minute level
    :return:
    """

    data = [
        ["2017-01-01 00:01:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 00:02:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 00:03:00", 300, 60, 0, 0, 1, 0],
        ["2017-01-01 00:04:00", 310, 50, 0, 0, 1, 0],
        ["2017-01-01 00:05:00", 320, 50, 0, 0, 1, 0],
        ["2017-01-01 00:06:00", 320, 50, 0, 0, 1, 0],
        ["2017-01-01 00:07:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 00:08:00", 0, 0, 0, 0, 0, 0],
        ["2017-01-01 00:09:00", 0.01, 0, 0, 0, 0, 0],
        ["2017-01-01 00:10:00", 0.1, 0, 0, 0, 0, 0],
        ["2017-01-01 00:11:00", 2.5, 0, 0, 0, 0, 0],
        ["2017-01-01 00:12:00", 10, 0, 0, 0, 0, 0],
        ["2017-01-01 00:13:00", 11.5, 0, 0, 0, 0, 0],
        ["2017-01-01 00:14:00", 12.5, 0, 0, 0, 0, 0],
        ["2017-01-01 00:15:00", 19, 0, 0, 0, 0, 0],
        ["2017-01-01 00:16:00", 50, 50, 0, 0, 1, 0],
        ["2017-01-01 00:17:00", 60, 10, 0, 0, 1, 0],
        ["2017-01-01 00:18:00", 70, 80, 0, 0, 1, 0],
        ["2017-01-01 00:19:00", 80, 10, 0, 0, 1, 0],
        ["2017-01-01 00:20:00", 90, 100, 0, 0, 1, 0],
        ["2017-01-01 00:21:00", 100, 100, 0, 0, 1, 0],
        ["2017-01-01 00:22:00", 110, 200, 0, 0, 1, 0],
        ["2017-01-01 00:23:00", 110, 300, 0, 0, 1, 0],
        ["2017-01-01 00:24:00", 110, 300, 200, 256, 1, 1],
        ["2017-01-01 00:25:00", 110, 300, 200, 256, 1, 1],
        ["2017-01-01 00:26:00", 180, 300, 6, 256, 1, 1],
        ["2017-01-01 00:27:00", 190, 300, 500, 256, 1, 1],
        ["2017-01-01 00:28:00", 200, 350, 500, 256, 1, 1],
        ["2017-01-01 00:29:00", 210, 360, 500, 256, 1, 1],
        ["2017-01-01 00:30:00", 250, 380, 500, 256, 1, 1],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "mill_a_power",
            "mill_a_load",
            "mill_b_power",
            "mill_b_load",
            "on_off_mill_a",
            "on_off_mill_b",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_mark_tags_off(
    sensor_with_on_off_df,
    tags_meta_config,
    tags_on_off_config,
    mark_tags_off_output_df,
):
    """
    Ensure dependent tags are marked as off (zero) if the corresponding on/off tag
    is off
    """

    output_df = set_off_equipment_to_zero(
        data=sensor_with_on_off_df,
        meta_config=tags_meta_config,
        on_off_dep_config=tags_on_off_config,
    )

    assert_frame_equal(output_df, mark_tags_off_output_df)


def test_on_off_tag_no_dependents(
    sensor_with_on_off_df,
    mark_tags_off_output_df,
    tags_meta_config,
    tags_on_off_config,
):
    """
    Ensure on_off tags can be defined without dependents
    """

    output_df = set_off_equipment_to_zero(
        data=sensor_with_on_off_df,
        meta_config=tags_meta_config,
        on_off_dep_config=tags_on_off_config,
    )

    assert_frame_equal(mark_tags_off_output_df, output_df)


def test_adj_on_off_minute_level(
    sensor_with_on_off_minute_level_df,
    tags_meta_config,
    tags_on_off_config,
    mark_tags_off_output_df_minute_level,
):
    """
    Ensure dependent tags are cleaned (set to zero) if the corresponding on/off
    tag is off. Minute level
    """

    output_df = set_off_equipment_to_zero(
        data=sensor_with_on_off_minute_level_df,
        meta_config=tags_meta_config,
        on_off_dep_config=tags_on_off_config,
    )

    assert_frame_equal(
        output_df,
        mark_tags_off_output_df_minute_level,
        check_dtype=False,
    )


def test_missing_on_off_tags_in_dict(
    sensor_with_on_off_df,
    tags_meta_config,
    tags_on_off_config,
    caplog,
):
    """
    Ensure error is raised when there are missing on off tags in the TagDict.
    """

    sensor_with_on_off_df.drop("on_off_mill_a", axis=1, inplace=True)
    sensor_with_on_off_df.drop("on_off_mill_b", axis=1, inplace=True)
    with caplog.at_level(logging.WARNING):
        set_off_equipment_to_zero(
            data=sensor_with_on_off_df,
            meta_config=tags_meta_config,
            on_off_dep_config=tags_on_off_config,
        )
    for record in caplog.records:
        assert record.levelname == "WARNING"

    assert (
        "There are no on/off tags defined in Tag Dictionary which match any "
        "of the columns in the supplied dataframe" in caplog.text
    )
