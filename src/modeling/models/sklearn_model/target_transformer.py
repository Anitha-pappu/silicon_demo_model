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


import typing as tp

import numpy as np
from pydantic import BaseModel
from sklearn.compose import TransformedTargetRegressor

from ... import utils
from ...types import Vector

_DEMO_ARRAY = np.random.rand(10)

TPotentialTransformer = tp.Callable[[Vector], tp.Any]


class TargetTransformerInitConfig(BaseModel):
    transformer: tp.Optional[utils.ObjectInitConfig] = None
    func: tp.Optional[str] = None
    inverse_func: tp.Optional[str] = None


def add_target_transformer(
    estimator: tp.Any,
    target_transformer: tp.Optional[utils.ObjectInitConfig] = None,
    func: tp.Optional[str] = None,
    inverse_func: tp.Optional[str] = None,
) -> TransformedTargetRegressor:
    """Wraps the given estimator in a TransformedTargetRegressor.

    TransformedTargetRegressor can be initialized using the transformer init config
    or using the pair of a ``func`` and ``inverse_func``.


    See documentation for ``TransformedTargetRegressor``:
        scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html

    Examples:

    Using the `target_transformer` object init config::

        target_transformer = {
            "class_name": "sklearn.preprocessing.MinMaxScaler",
            "kwargs": {
                "feature_range": [0,2],
            }
        }
        add_target_transformer(
            LinearRegression(),
            target_transformer=ObjectInitConfig(**target_transformer),
        )

    Using `func` and `inverse_func` pair::

        add_target_transformer(
            LinearRegression(),
            func="numpy.log1p",
            inverse_func="numpy.expm1",
        )

    Returns:
        ``sklearn.compose.TransformedTargetRegressor`` with wrapped estimator

    Raises:
        `InvalidTransformerFuncError`: if supplied transformation function
        is not valid for this purpose.
    """
    if target_transformer is not None:
        transformer = utils.load_obj(target_transformer.class_name)(
            **target_transformer.kwargs,
        )
        return TransformedTargetRegressor(regressor=estimator, transformer=transformer)

    if func is not None and inverse_func is not None:
        func_parsed: TPotentialTransformer = utils.load_obj(func)
        inverse_func_parsed: TPotentialTransformer = utils.load_obj(inverse_func)

        both_valid = (
            _check_is_valid_target_transformer(func_parsed)
            and _check_is_valid_target_transformer(inverse_func_parsed)
        )
        if not both_valid:
            raise InvalidTransformerFuncError(
                "Either func or inverse func supplied aren't valid transformers.",
            )

        return TransformedTargetRegressor(
            regressor=estimator,
            func=func_parsed,
            inverse_func=inverse_func_parsed,
        )

    raise ValueError(
        "Target transformer can't be initialised with provided "
        "target transformer init config is not",
    )


def _check_is_valid_target_transformer(func: TPotentialTransformer) -> bool:
    """Checks if a function is a valid target transformer.

    Args:
        func: to test on validity.

    Returns:
        True if valid, False otherwise.
    """

    requirements = (
        _check_preserves_length(func),
        _check_produces_numbers(func),
    )
    return all(requirements)


def _check_preserves_length(func: TPotentialTransformer) -> bool:
    """Checks if a potential target transformer function
    preserves input length.

    Args:
        func: to test on length consistency.

    Returns:
        True if preserves length, False otherwise.
    """
    transformed = func(_DEMO_ARRAY)
    try:
        return len(transformed) == len(_DEMO_ARRAY)
    except TypeError:
        return False


def _check_produces_numbers(func: TPotentialTransformer) -> bool:
    """Checks if a potential target transformer function
    produces numeric outputs.

    Args:
        func: to test on output type.

    Returns:
        True if produces numbers, False otherwise.
    """
    if not _check_preserves_length(func):
        return False

    transformed = func(_DEMO_ARRAY)
    return all(isinstance(element, float) for element in transformed)


class InvalidTransformerFuncError(Exception):
    """Raised if target transformer function is invalid."""