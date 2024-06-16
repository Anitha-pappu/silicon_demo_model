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
Validation utility functions.
"""
import typing as tp
from functools import partial, update_wrapper

import numpy as np
import pandas as pd

from optimizer.types import (
    MapsMatrixToMatrix,
    Matrix,
    ReducesMatrixToSeries,
    Vector,
)

_T = tp.TypeVar("_T")


def check_matrix(
    data: tp.Union[Matrix, Vector],
    expected_shape: tp.Optional[tp.Tuple[int, ...]] = None,
) -> None:
    """Ensure the provided data is 2-D and throw a helpful error if not.

    Args:
        data: data to check for dimensionality.
        expected_shape: optional shape to check for.

    Raises:
        ValueError: when the provided data is one dimensional.
    """
    # Source: sklearn.utils.validation.check_array
    if isinstance(data, pd.Series):
        raise ValueError(
            f"Expected DataFrame, got a Series instead: \nseries={data}.\n"
            f"Convert your data to a DataFrame using pd.DataFrame(series) if "
            f"your data has a single feature or series.to_frame().T "
            f"if it contains a single sample."
        )

    if isinstance(data, np.ndarray):
        if data.ndim in {0, 1}:
            raise ValueError(
                f"Expected 2D array, got scalar array instead:\narray={data}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                f"your data has a single feature or array.reshape(1, -1) "
                f"if it contains a single sample with multiple features."
            )

        if data.ndim >= 3:
            raise ValueError(f"Found array with dim {data.ndim}. Expected <= 2.")

    if expected_shape is not None:
        if data.shape != expected_shape:
            raise ValueError(
                f"Expected 2D Matrix of shape {expected_shape}, "
                f"got incorrect shape {data.shape}"
            )


def check_vector(
    data: tp.Union[Matrix, Vector],
    expected_length: tp.Optional[int] = None,
) -> None:
    """Ensure the provided data is 1-D and throw a helpful error if not.

    Args:
        data: data to check for dimensionality.
        expected_length: optional length to check for.

    Raises:
        ValueError: when the provided data is not one dimensional.
    """
    if data.squeeze().ndim > 1:  # Works for Pandas and Numpy.
        raise ValueError(
            f"Expected 1D Vector, got {type(data).__name__} with shape {data.shape}"
        )
    if expected_length is not None:
        if data.size != expected_length:
            raise ValueError(
                f"Expected 1D Vector of length {expected_length}, "
                f"got vector with length {data.size}"
            )


def wrap_with_reducer_dim_check(
    reducer: ReducesMatrixToSeries,
) -> ReducesMatrixToSeries:
    reducer_with_check = partial(reduce_with_dim_check, reducer=reducer)
    return update_wrapper(reducer_with_check, reducer)


def wrap_with_mapper_dim_check(
    mapper: MapsMatrixToMatrix,
) -> MapsMatrixToMatrix:
    mapper_with_check = partial(map_with_dim_check, mapper=mapper)
    return update_wrapper(mapper_with_check, mapper)


def reduce_with_dim_check(
    reducer_input: Matrix, reducer: ReducesMatrixToSeries,
) -> Vector:
    """
    Applies reducer and checks that reducer actually reduces the 2nd dim

    Returns:
        ``reducer(reducer_input)``
    """
    reducer_result = reducer(reducer_input)
    check_vector(reducer_result, expected_length=reducer_input.shape[0])
    return reducer_result


def map_with_dim_check(
    mapper_input: Matrix, mapper: MapsMatrixToMatrix,
) -> Vector:
    """
    Applies reducer and checks that reducer actually reduces the 2nd dim

    Returns:
        ``mapper(mapper_input)``
    """
    mapper_result = mapper(mapper_input)
    check_matrix(mapper_result, expected_shape=mapper_input.shape)
    return mapper_result


class DimCheckingWrapper(tp.Generic[_T]):
    def __init__(
        self,
        callable_object: tp.Callable[[Matrix, tp.Any], _T],
        expected_output_type: _T,
    ) -> None:
        self.callable_object = callable_object
        self._expected_output_type = expected_output_type
        if expected_output_type not in {Matrix, Vector}:
            raise ValueError(f"Unknown {expected_output_type = }")

    def __call__(self, parameters: Matrix, *args: tp.Any, **kwargs: tp.Any) -> _T:
        result = self.callable_object(parameters, *args, **kwargs)
        if self._expected_output_type is Matrix:
            check_matrix(result, expected_shape=parameters.shape)
        check_vector(result, expected_length=parameters.shape[0])
        return result
