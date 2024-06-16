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
Functional utilities.
"""

import typing as tp
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd

from optimizer.types import (
    HasName,
    Matrix,
    Predictor,
    ReducesMatrixToSeries,
    Vector,
)


class _Chainer:
    """
    Class to hold a list of arguments given to chain.
    This avoids the issue that the result of chain will not be pickleable.
    """

    def __init__(
        self,
        *args: tp.List[tp.Union[tp.Callable[[Matrix], Vector], Predictor]],
    ) -> None:
        """Constructor.

        Args:
            args: argument list.
        """
        self.args = args

    def __call__(self, x: tp.Any) -> tp.Any:
        """Call the argument callables / predictors in order.

        Args:
            x: input argument to the composed functions.

        Returns:
            Output of the composed functions.
        """
        out = x

        for f in self.args:
            if isinstance(f, Predictor):
                out = f.predict(out)
            else:
                out = f(out)  # type: ignore

        return out


def chain(
    *args: tp.List[tp.Union[tp.Callable[[Matrix], Vector], Predictor]],
) -> tp.Callable[[tp.Any], tp.Any]:
    """Chain a list of single argument callables and/or models.

    Calls the first, then second, and so on through the provided arguments.

    Args:
        args: list of callables.

    Returns:
        Callable.
    """
    return _Chainer(*args)


def safe_subset(
    parameters: Matrix, col: tp.Union[str, int, tp.Sequence[tp.Union[str, int]]],
) -> tp.Union[Matrix, Vector]:
    """
    Safely index into a parameter matrix based off its type.

    Args:
        parameters: Matrix to index into.
        col: str or int, column specifier.

    Returns:
        Matrix or vector, the subset parameters matrix.
    """
    result = (
        parameters[:, col]  # type: ignore
        if isinstance(parameters, np.ndarray)
        else parameters[col]
    )
    result_casted = (
        tp.cast(Matrix, result)
        if isinstance(col, list)
        else tp.cast(Vector, result)
    )
    return result_casted


def safe_assign_column(
    parameters: Matrix,
    values: Vector,
    col: tp.Union[str, int, slice],
    row_mask: tp.Union[Vector, tp.List[tp.Any]] = None,
) -> Matrix:
    """Safely assigns a column of a matrix to the provided values.

    Args:
        parameters: Matrix to assign values to.
        values: values to assign.
        col: which column to put values into.
        row_mask: optional row mask.
            Must be a boolean type with len(row_mask) == parameter.shape[1].

    Returns:
        Matrix, with the updated column.

    Raises:
        ValueError: if values is 1 dimensional.
        ValueError: if the provided row_mask is a different length and values.
    """
    values = np.array(values)

    if values.ndim != 1:
        raise ValueError("Values must be a Vector.")

    if row_mask is not None and len(row_mask) != len(values):
        raise ValueError("Length and shape of row_mask must be the same as values.")

    parameters_copy = deepcopy(parameters)

    if isinstance(parameters_copy, np.ndarray):
        row_mask = slice(None) if row_mask is None else row_mask
        parameters_copy[row_mask, col] = values[row_mask]
        return parameters_copy
    elif row_mask is None:
        parameters_copy[col] = values
        return parameters_copy
    else:
        # This fixed a problem caused by DatetimeIndex.
        series_row_mask: pd.Series = pd.Series(row_mask)
        series_row_mask.index = parameters_copy.index
        parameters_copy.loc[series_row_mask, col] = values[series_row_mask]
        return parameters_copy


def safe_assign_rows(
    assignee: Matrix, assignment: Matrix, row_mask: tp.Union[Vector, tp.List[tp.Any]],
) -> Matrix:
    """Safely assigns a rows of one matrix to be the rows of the other.

    Args:
        assignee: Matrix to assign values to.
        assignment: values to assign.
        row_mask: boolean mask, True rows are assigned a new value in assignee

    Returns:
        Matrix, with the updated rows.

    Raises:
        ValueError: if assignee and assignment shapes are different.
        ValueError: if the length row_mask is not equal to assignee rows.
    """
    if assignee.shape != assignment.shape:
        raise ValueError(
            f"Shapes of matrices must be equal. "
            f"Got {assignee.shape} and {assignment.shape}."
        )

    if assignee.shape[0] != len(row_mask):
        raise ValueError(
            f"Length of row mask must be equal to rows in provided matrices. "
            f"Got length {len(row_mask)}. Expected length {assignee.shape[0]}"
        )

    assignment = np.array(assignment)
    assignee = deepcopy(assignee)

    if isinstance(assignee, np.ndarray):
        assignee[row_mask, :] = assignment[row_mask, :]
        return assignee
    # This fixed a problem caused by DatetimeIndex.
    series_row_mask = pd.Series(row_mask)
    series_row_mask.index = assignee.index
    assignee.loc[series_row_mask, :] = assignment[series_row_mask]
    return assignee


def column(col: tp.Union[str, float]) -> ReducesMatrixToSeries:
    """Returns a function that will handle indexing a given Matrix.
    This function is used to get around using lambdas in constraint definitions that
    would not allow pickling.

    Args:
        col: string or integer column name.

    Returns:
        Callable.
    """
    return partial(safe_subset, col=col)


def get_penalty_name(penalty: HasName, default_name: str = "penalty") -> str:
    return penalty.name or default_name
