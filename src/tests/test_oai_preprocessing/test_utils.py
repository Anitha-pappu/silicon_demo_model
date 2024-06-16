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
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from preprocessing import TagMetaParameters, TagsConfig
from preprocessing.utils import (
    _calculate_clipped_values,
    count_outlier,
    count_outside_threshold,
    create_range_map,
    get_clip_summary,
    get_drop_summary,
)


def test__calculate_clipped_values():
    series_with_outliers = pd.Series([1, 2, 3, 4, 5, np.nan])
    lower = 2
    upper = 4
    expected_count = 2

    calculated_count = _calculate_clipped_values(series_with_outliers, lower, upper)

    assert calculated_count == expected_count


def test_get_clip_summary():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [3, 3, 3, np.nan, 3],
    })

    tag_range = {
        'A': (2, 4),  # A has two outliers: 1 and 5
        'B': (2, 4),  # B has two outliers: 5 and 1
        'C': (1, 5),  # C has no outliers
    }

    expected_summary = pd.DataFrame({
        'tag_name': ['A', 'B'],
        'outlier_percentage': [40.0, 40.0],
    })

    summary_df = get_clip_summary(data, tag_range)

    assert_frame_equal(
        summary_df.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
    )


def test_get_drop_summary_no_outliers_dropped():
    original_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
    })
    cleaned_data = original_data.copy()  # No outliers dropped

    expected_summary = pd.DataFrame(columns=["tag_name", "outlier_percentage"])

    summary_df = get_drop_summary(original_data, cleaned_data)

    assert_frame_equal(
        summary_df,
        expected_summary,
        check_dtype=False,
        check_index_type=False,
    )


def test_get_drop_summary_all_outliers_dropped():
    original_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
    })
    cleaned_data = original_data.copy()
    cleaned_data.loc[1, 'A'] = None  # All values in 'A' are dropped

    expected_summary = pd.DataFrame({
        "tag_name": ['A'],
        "outlier_percentage": [33.33],  # 33.33% of 'A' data is dropped
    })

    summary_df = get_drop_summary(original_data, cleaned_data)

    assert_frame_equal(summary_df, expected_summary)


def test_get_drop_summary_some_outliers_dropped():
    original_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9],
    })
    cleaned_data = original_data.copy()
    cleaned_data.loc[0, 'A'] = None  # One outlier dropped in 'A'
    cleaned_data.loc[2, 'B'] = None  # One outlier dropped in 'B'

    expected_summary = pd.DataFrame({
        "tag_name": ['A', 'B'],
        "outlier_percentage": [33.33, 33.33],
    })

    summary_df = get_drop_summary(original_data, cleaned_data)

    assert_frame_equal(
        summary_df.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
        check_index_type=False,
    )


def test_count_outliers_lower():
    series = pd.Series([1, 2, 3, 4, 5])
    threshold = 3
    count = count_outside_threshold(series, threshold, "lower")
    assert count == 2


def test_count_outliers_upper():
    series = pd.Series([1, 2, 3, 4, 5])
    threshold = 3
    count = count_outside_threshold(series, threshold, "upper")
    assert count == 2


def test_count_outliers_default_direction():
    series = pd.Series([1, 2, 3, 4, 5])
    threshold = 3
    count = count_outside_threshold(series, threshold)
    assert count == 2


def test_count_outliers_invalid_direction():
    series = pd.Series([1, 2, 3, 4, 5])
    threshold = 3
    with pytest.raises(ValueError):
        count_outside_threshold(series, threshold, "invalid")


def test_count_outlier_with_meta_config():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 50],
        'B': [6, 7, 8, 9, -20],
    })
    tags_meta_config_data = [
        TagMetaParameters(
            tag_name='A',
            range_min=2,
            range_max=4,
            data_type='numeric',
            tag_type='input',
        ),
        TagMetaParameters(
            tag_name='B',
            range_min=6,
            range_max=9,
            data_type='numeric',
            tag_type='input',
        ),
    ]
    tags_meta_config = TagsConfig(
        tags_parameters=tags_meta_config_data,
        model_schema=TagMetaParameters,
    )

    expected_df = pd.DataFrame(
        {
            'below_range_min_count': [1, 1],
            'above_range_max_count': [1, 1],
        },
        index=['A', 'B'],
    )

    outliers_df = count_outlier(data, tags_meta_config)

    assert_frame_equal(outliers_df, expected_df, check_dtype=False)


def test_count_outlier_without_meta_config():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 50],
        'B': [6, 7, 8, 9, -20],
    })

    # Assuming no outliers on the 5th and 95th percentile boundaries
    expected_df = pd.DataFrame(
        {
            'below_range_min_count': [1, 1],
            'above_range_max_count': [1, 1],
        },
        index=['A', 'B'],
    )

    outliers_df = count_outlier(data)

    assert_frame_equal(outliers_df, expected_df, check_dtype=False)


def test_create_range_map_with_valid_values():
    tags_meta_config_data = [
        TagMetaParameters(
            tag_name='A',
            min=1.0,
            max=5.0,
            data_type='numeric',
            tag_type='input',
        ),
        TagMetaParameters(
            tag_name='B',
            min=2.0,
            max=6.0,
            data_type='numeric',
            tag_type='input',
        ),
    ]
    tags_meta_config = TagsConfig(
        tags_parameters=tags_meta_config_data,
        model_schema=TagMetaParameters,
    )

    expected_range_map = {
        'A': (1.0, 5.0),
        'B': (2.0, 6.0),
    }

    range_map = create_range_map(tags_meta_config)
    assert range_map == expected_range_map


def test_create_range_map_ignores_none_values():
    tags_meta_config_data = [
        TagMetaParameters(
            tag_name='A',
            min=None,
            max=5.0,
            data_type='numeric',
            tag_type='input',
        ),
        TagMetaParameters(
            tag_name='B',
            min=2.0,
            max=None,
            data_type='numeric',
            tag_type='input',
        ),
        TagMetaParameters(
            tag_name='C',
            min=None,
            max=None,
            data_type='numeric',
            tag_type='input',
        ),
    ]
    tags_meta_config = TagsConfig(
        tags_parameters=tags_meta_config_data,
        model_schema=TagMetaParameters,
    )

    # Expected range_map should be empty since all tags have either min or max as None
    expected_range_map = {}

    range_map = create_range_map(tags_meta_config)
    assert range_map == expected_range_map
