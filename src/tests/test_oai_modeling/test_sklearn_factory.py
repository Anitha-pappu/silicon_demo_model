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


import numpy as np
import pandas as pd
import pytest

from modeling import SklearnPipelineFactory

N_FEATURES = 10


@pytest.fixture(scope="function")
def sklearn_pipeline_init_config_no_wrapper():
    return {
        "transformers": [
            {
                "name": "poly_transformer",
                "class_name": "sklearn.preprocessing.PolynomialFeatures",
                "kwargs": {
                    "degree": 2,
                    "include_bias": False,
                    "interaction_only": False,
                },
                "wrapper": None,
            },
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": None,
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


@pytest.fixture(scope="function")
def sklearn_pipeline_init_config_preserve_pandas_wrapper():
    return {
        "transformers": [
            {
                "name": "poly_transformer",
                "class_name": "sklearn.preprocessing.PolynomialFeatures",
                "kwargs": {
                    "degree": 2,
                    "include_bias": False,
                    "interaction_only": False,
                },
                "wrapper": "preserve_pandas",
            },
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": "preserve_pandas",
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


@pytest.fixture(scope="function")
def sklearn_pipeline_init_config_preserve_columns_wrapper():
    return {
        "transformers": [
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": "preserve_columns",
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


@pytest.fixture(scope="function")
def sklearn_pipeline_init_config_failing():
    return {
        "transformers": [
            {
                "name": "poly_transformer",
                "class_name": "sklearn.preprocessing.PolynomialFeatures",
                "kwargs": {
                    "degree": 2,
                    "include_bias": False,
                    "interaction_only": False,
                },
                "wrapper": "preserve_columns",
            },
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": "preserve_pandas",
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


@pytest.fixture(scope="function")
def sklearn_pipeline_init_select_columns():
    return {
        "transformers": [
            {
                "name": "poly_transformer",
                "class_name": "sklearn.feature_selection.SelectKBest",
                "kwargs": {
                    "k": 3,
                    "score_func": "sklearn.feature_selection.f_regression",
                },
                "wrapper": "select_columns",
            },
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": "preserve_columns",
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


def test_sklearn_pipeline_factory_without_wrapping_transformers(
    regression_data,
    sklearn_pipeline_init_config_no_wrapper,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_config_no_wrapper,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    sklearn_pipeline = sklearn_pipeline_factory.create().fit(regression_data)
    transformed_data = sklearn_pipeline.transform(regression_data)
    assert sklearn_pipeline.features_in != sklearn_pipeline.features_out
    assert len(sklearn_pipeline.features_out) == (
        N_FEATURES * (N_FEATURES - 1) / 2 + 2 * N_FEATURES
    )
    assert transformed_data.shape == (
        len(regression_data),
        len(sklearn_pipeline.features_out),
    )
    assert isinstance(transformed_data, np.ndarray)


def test_sklearn_pipeline_factory_with_preserve_pandas_wrapper(
    regression_data,
    sklearn_pipeline_init_config_preserve_pandas_wrapper,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_config_preserve_pandas_wrapper,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    sklearn_pipeline = sklearn_pipeline_factory.create().fit(regression_data)
    transformed_data = sklearn_pipeline.transform(regression_data)
    assert sklearn_pipeline.features_in != sklearn_pipeline.features_out
    assert len(sklearn_pipeline.features_out) == (
        N_FEATURES * (N_FEATURES - 1) / 2 + 2 * N_FEATURES
    )
    assert transformed_data.shape == (
        len(regression_data),
        len(sklearn_pipeline.features_out),
    )
    assert isinstance(transformed_data, pd.DataFrame)


def test_sklearn_pipeline_factory_with_preserve_columns_wrapper(
    regression_data,
    sklearn_pipeline_init_config_preserve_columns_wrapper,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_config_preserve_columns_wrapper,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    sklearn_pipeline = sklearn_pipeline_factory.create().fit(regression_data)
    transformed_data = sklearn_pipeline.transform(regression_data)
    assert sklearn_pipeline.features_in == sklearn_pipeline.features_out
    assert transformed_data.shape == (
        len(regression_data),
        len(sklearn_pipeline.features_out),
    )
    assert isinstance(transformed_data, pd.DataFrame)


def test_sklearn_pipeline_factory_fails_preserve_columns_and_col_count_is_changed(
    regression_data,
    sklearn_pipeline_init_config_failing,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_config_failing,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    with pytest.raises(RuntimeError):
        sklearn_pipeline_factory.create().fit(regression_data)


def test_sklearn_pipeline_factory_with_select_solumns_wrapper(
    regression_data,
    sklearn_pipeline_init_select_columns,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_select_columns,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    sklearn_pipeline = sklearn_pipeline_factory.create().fit(regression_data)
    transformed_data = sklearn_pipeline.transform(regression_data)
    assert len(sklearn_pipeline.features_out) == 3
    assert transformed_data.shape == (
        len(regression_data),
        len(sklearn_pipeline.features_out),
    )
    assert isinstance(transformed_data, pd.DataFrame)
