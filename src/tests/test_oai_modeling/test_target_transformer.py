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
Tests for model/pipeline initialization involving target transformation.
"""
import typing as tp
from types import MappingProxyType

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor

from modeling import SklearnModelFactory, SklearnPipelineFactory
from modeling.models.sklearn_model.target_transformer import (
    TPotentialTransformer,
    _check_is_valid_target_transformer,
)

_DUMMY_FEATURES_IN = ("first", "second")
_DUMMY_TARGET = "target"

_LINEAR_INIT_CONFIG_WITH_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.linear_model.LinearRegression",
        },
        "target_transformer": {
            "transformer": {
                "class_name": "sklearn.preprocessing.MinMaxScaler",
                "kwargs": {
                    "feature_range": [0, 2],
                },
            },
        },
    },
)

_LINEAR_INIT_CONFIG_WITHOUT_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.linear_model.LinearRegression",
        },
    },
)

_TREE_INIT_CONFIG_WITH_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.tree.DecisionTreeRegressor",
        },
        "target_transformer": {
            "transformer": {
                "class_name": "sklearn.preprocessing.MinMaxScaler",
                "kwargs": {},
            },
        },
    },
)

_TREE_INIT_CONFIG_WITHOUT_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.tree.DecisionTreeRegressor",
        },
    },
)

_PIPELINE_INIT_CONFIG_WITH_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.tree.DecisionTreeRegressor",
        },
        "transformers": [
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
            },
        ],
        "target_transformer": {
            "transformer": {
                "class_name": "sklearn.preprocessing.MinMaxScaler",
                "kwargs": {},
            },
        },
    },
)

_PIPELINE_INIT_CONFIG_WITHOUT_TRANSFORMER = MappingProxyType(
    {
        "estimator": {
            "class_name": "sklearn.tree.DecisionTreeRegressor",
        },
        "transformers": [
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
            },
        ],
    },
)


@pytest.mark.parametrize(
    "model_init_config,should_be_transformed",
    [
        (_LINEAR_INIT_CONFIG_WITH_TRANSFORMER, True),
        (_TREE_INIT_CONFIG_WITH_TRANSFORMER, True),
        (_LINEAR_INIT_CONFIG_WITHOUT_TRANSFORMER, False),
        (_TREE_INIT_CONFIG_WITHOUT_TRANSFORMER, False),
    ],
)
def test_model_factory_creates_transformed_target_regressor_when_needed(
    model_init_config: tp.Dict[str, tp.Any],
    should_be_transformed: bool,
) -> None:
    """Test that transformed target case is initialized properly
    via sklearn model factory."""

    model_factory = SklearnModelFactory(
        model_init_config=model_init_config,
        features_in=_DUMMY_FEATURES_IN,
        target=_DUMMY_TARGET,
    )
    model = model_factory.create()

    assert (
        isinstance(model.estimator, TransformedTargetRegressor) == should_be_transformed
    )


@pytest.mark.parametrize(
    "pipeline_init_config,should_be_transformed",
    [
        (_PIPELINE_INIT_CONFIG_WITH_TRANSFORMER, True),
        (_PIPELINE_INIT_CONFIG_WITHOUT_TRANSFORMER, False),
    ],
)
def test_pipeline_factory_creates_transformed_target_regressor_when_needed(
    pipeline_init_config: tp.Dict[str, tp.Any],
    should_be_transformed: bool,
) -> None:
    """Test that transformed target case is initialized properly
    via sklearn pipeline factory."""

    pipeline_factory = SklearnPipelineFactory(
        model_init_config=pipeline_init_config,
        features_in=_DUMMY_FEATURES_IN,
        target=_DUMMY_TARGET,
    )
    pipeline = pipeline_factory.create()

    assert (
        isinstance(pipeline.estimator, TransformedTargetRegressor)
        == should_be_transformed
    )


@pytest.mark.parametrize(
    "func,expectation",
    [
        (np.sum, False),
        (np.product, False),
        (np.square, True),
        (np.sqrt, True),
        (np.log10, True),
    ],
)
def test_check_is_valid_target_transformer(
    func: TPotentialTransformer,
    expectation: bool,
) -> None:
    """Test that target transformer validity is assessed properly."""
    output = _check_is_valid_target_transformer(func)
    assert output == expectation
