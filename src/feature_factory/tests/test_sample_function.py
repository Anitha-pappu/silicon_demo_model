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
Tests sample_function.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from feature_factory.sample_function import (
    pandas_cut,
    pandas_divide,
    pandas_max,
    pandas_mean,
    pandas_min,
    pandas_prod,
    pandas_qcut,
    pandas_subtract,
    pandas_sum,
)


@pytest.fixture
def df_input():
    return pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [7, 8, 9]})


@pytest.fixture
def df_input_with_null():
    return pd.DataFrame({"foo": [1, 2, np.nan], "bar": [4, 5, 6], "baz": [7, 8, 9]})


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_subtract(df_input, dependencies):

    result = pandas_subtract(df_input, dependencies)
    expected = pd.Series([-3, -3, -3])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_subtract_with_null(df_input_with_null, dependencies):

    result = pandas_subtract(df_input_with_null, dependencies)
    expected = pd.Series([-3, -3, np.nan])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_divide(df_input, dependencies):

    result = pandas_divide(df_input, dependencies)
    expected = pd.Series([1 / 4, 2 / 5, 3 / 6])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_divide_with_null(df_input_with_null, dependencies):

    result = pandas_divide(df_input_with_null, dependencies)
    expected = pd.Series([1 / 4, 2 / 5, np.nan])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_mean(df_input, dependencies):

    result = pandas_mean(df_input, dependencies)
    expected = pd.Series([2.5, 3.5, 4.5])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_mean_with_null(df_input_with_null, dependencies):

    result = pandas_mean(df_input_with_null, dependencies)
    expected = pd.Series([2.5, 3.5, 6.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_prod(df_input, dependencies):

    result = pandas_prod(df_input, dependencies)
    expected = pd.Series([4, 10, 18])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar"]])
def test_pandas_prod_with_null(df_input_with_null, dependencies):

    result = pandas_prod(df_input_with_null, dependencies)
    expected = pd.Series([4.0, 10.0, 6.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_max(df_input, dependencies):

    result = pandas_max(df_input, dependencies)
    expected = pd.Series([7, 8, 9])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_max_with_null(df_input_with_null, dependencies):

    result = pandas_max(df_input_with_null, dependencies)
    expected = pd.Series([7.0, 8.0, 9.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_sum(df_input, dependencies):

    result = pandas_sum(df_input, dependencies)
    expected = pd.Series([12, 15, 18])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_sum_with_null(df_input_with_null, dependencies):

    result = pandas_sum(df_input_with_null, dependencies)
    expected = pd.Series([12.0, 15.0, 15.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_min(df_input, dependencies):

    result = pandas_min(df_input, dependencies)
    expected = pd.Series([1, 2, 3])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies", [["foo", "bar", "baz"]])
def test_pandas_min_with_null(df_input_with_null, dependencies):

    result = pandas_min(df_input_with_null, dependencies)
    expected = pd.Series([1.0, 2.0, 6.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo"], {"q": 2})])
def test_pandas_qcut(df_input, dependencies, kwargs):

    result = pandas_qcut(df_input, dependencies, **kwargs)
    expected = pd.Series(
        [
            pd._libs.interval.Interval(0.999, 2.0, closed='right'),
            pd._libs.interval.Interval(0.999, 2.0, closed='right'),
            pd._libs.interval.Interval(2.0, 3.0, closed='right'),
        ],
        dtype=pd.CategoricalDtype(ordered=True),
        name="foo",
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo"], {"q": 2})])
def test_pandas_qcut_with_null(df_input_with_null, dependencies, kwargs):

    result = pandas_qcut(df_input_with_null, dependencies, **kwargs)
    expected = pd.Series(
        [
            pd._libs.interval.Interval(0.999, 1.5, closed='right'),
            pd._libs.interval.Interval(1.5, 2.0, closed='right'),
            np.nan,
        ],
        dtype=pd.CategoricalDtype(ordered=True),
        name="foo",
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo", "bar"], {"q": 2})])
def test_pandas_qcut_value_error(df_input, dependencies, kwargs):
    with pytest.raises(ValueError):
        pandas_qcut(df_input, dependencies, **kwargs)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo"], {"bins": 2})])
def test_pandas_cut(df_input, dependencies, kwargs):

    result = pandas_cut(df_input, dependencies, **kwargs)
    expected = pd.Series(
        [
            pd._libs.interval.Interval(0.998, 2.0, closed='right'),
            pd._libs.interval.Interval(0.998, 2.0, closed='right'),
            pd._libs.interval.Interval(2.0, 3.0, closed='right'),
        ],
        dtype=pd.CategoricalDtype(ordered=True),
        name="foo",
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo"], {"bins": 2})])
def test_pandas_cut_with_null(df_input_with_null, dependencies, kwargs):

    result = pandas_cut(df_input_with_null, dependencies, **kwargs)
    expected = pd.Series(
        [
            pd._libs.interval.Interval(0.999, 1.5, closed='right'),
            pd._libs.interval.Interval(1.5, 2.0, closed='right'),
            np.nan,
        ],
        dtype=pd.CategoricalDtype(ordered=True),
        name="foo",
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dependencies, kwargs", [(["foo", "bar"], {"bins": 2})])
def test_pandas_cut_value_error(df_input, dependencies, kwargs):
    with pytest.raises(ValueError):
        pandas_cut(df_input, dependencies, **kwargs)
