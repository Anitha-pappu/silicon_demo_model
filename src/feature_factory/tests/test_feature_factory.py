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
Tests DerivedFeaturesMaker transformer.
"""
import pickle

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError

from feature_factory import FeatureFactory
from feature_factory.feature_factory import FunctionReturnError, NotDagError


@pytest.fixture
def df():
    return pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [7, 8, 9]})


def mean_func_return_series(df, cols):
    return df[cols].mean(1)


def mean_func_return_df(df, cols, derived_feature="mean_foo_bar"):
    ans = df.copy()
    ans[derived_feature] = ans[cols].mean(1)
    return ans


def prod_func(df, cols):
    return df[cols].prod(1)


def int_func(df, cols):
    return 1


def wrong_df_func(df, cols):
    return df.rename(columns=dict(zip(df, "abc")))


def params(func_a="mean_func_return_series", func_b="prod_func"):
    return {
        "mean_foo_bar": {
            "dependencies": ["foo", "bar"],
            "function": f"{__name__}.{func_a}",
        },
        "prod_mean_baz": {
            "dependencies": ["mean_foo_bar", "baz"],
            "function": f"{__name__}.{func_b}",
        },
    }


@pytest.fixture
def not_a_dag_params(func_a="mean_func_return_series", func_b="prod_func"):
    return {
        "mean_foo_bar": {
            "dependencies": ["prod_mean_baz", "bar"],
            "function": f"{__name__}.{func_a}",
        },
        "prod_mean_baz": {
            "dependencies": ["mean_foo_bar", "baz"],
            "function": f"{__name__}.{func_b}",
        },
    }


@pytest.fixture
def default_correct_params():
    return params()


# pylint : disable=redefined-outer-name
@pytest.mark.parametrize("func_a", ["mean_func_return_series", "mean_func_return_df"])
def test_df_eval_success(df, func_a, func_b="prod_func"):
    params_correct = params(func_a, func_b)
    orig_df = df.copy()
    transformer = FeatureFactory(params_correct)
    transformed = transformer.fit_transform(df)
    orig_df["mean_foo_bar"] = orig_df[["foo", "bar"]].mean(1)
    orig_df["prod_mean_baz"] = orig_df[["mean_foo_bar", "baz"]].prod(1)
    assert_frame_equal(transformed, orig_df)


@pytest.mark.parametrize("func_a", ["wrong_df_func", "int_func"])
def test_raise_func_error_with_wrong_return_type(df, func_a, func_b="prod_func"):
    params_incorrect = params(func_a, func_b)
    transformer = FeatureFactory(params_incorrect)
    with pytest.raises(FunctionReturnError, match="The function of"):
        transformer.fit_transform(df)


def test_raise_value_error_with_leaf_not_in_df(df, default_correct_params):
    df = df.drop(columns=["baz"])
    transformer = FeatureFactory(default_correct_params)
    with pytest.raises(ValueError, match="must be present in "):
        transformer.fit_transform(df)


def test_raise_not_a_dag_error(not_a_dag_params):
    with pytest.raises(NotDagError, match="graph of eng"):
        FeatureFactory(not_a_dag_params)


def test_no_transform_without_fit(df, default_correct_params):
    transformer = FeatureFactory(default_correct_params)
    with pytest.raises(NotFittedError):
        transformer.transform(df)


def test_transformer_is_serializable(df, default_correct_params):
    transformer = FeatureFactory(default_correct_params)
    transformer.fit_transform(df)
    pickle.loads(pickle.dumps(transformer))
