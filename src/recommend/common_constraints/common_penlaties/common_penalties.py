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

import pandas as pd

from optimizer import Penalty
from optimizer import penalty as create_penalty

from ...utils import validate_is_a_single_row_dataframe


def create_similar_delta_penalty(
    row_to_optimise: pd.DataFrame,
    controls: tp.List[str],
    penalty_multiplier: float = 1.0,
) -> tp.List[Penalty]:
    """
    Creates constraint that aligns two or more controls' deltas.

    I.e. creates such penalties that make sure
    ``controls`` have similar recommended delta = initial value - recommended value.

    Implementation details:
        First control from ``controls`` is considered a reference control.
        All deltas are aligned against reference control's delta.
        Hence, number of penalties is ``len(controls)-1`` and each penalty
        is expressed as control delta "==" reference control delta.

    Example:
        Imagine three controls "A", "B", "C". Then we produce two penalties:
        delta("A") == delta("B") and delta("A") == delta("C").
    """
    validate_is_a_single_row_dataframe(row_to_optimise)

    if len(controls) < 2:
        raise ValueError("Please provide at least two controls")

    return [
        create_penalty(
            _DeltaEvaluator(control, row_to_optimise),
            "==",
            _DeltaEvaluator(controls[0], row_to_optimise),
            penalty_multiplier=penalty_multiplier,
        )
        for control in controls[1:]
    ]


def create_similar_value_penalty(
    controls: tp.List[str],
    penalty_multiplier: float = 1.0,
) -> tp.List[Penalty]:
    """
    Creates constraint that aligns two or more controls'.

    I.e. creates such penalties that make sure ``controls``
    have similar recommended values.

    Implementation details:
        First control from ``controls`` is considered a reference control.
        All other controls are aligned against reference control.
        Hence, number of penalties is ``len(controls)-1``
        and each penalty is expressed as control "==" reference control.

    Example:
        Imagine three controls "A", "B", "C". Then we produce two penalties:
        "A" == "B" and "A" == "C".
    """
    if len(controls) < 2:
        raise ValueError("Please provide at least two controls")

    return [
        create_penalty(
            control, "==", controls[0], penalty_multiplier=penalty_multiplier,
        )
        for control in controls[1:]
    ]


class _DeltaEvaluator(object):
    """When called evaluates difference between current column's value and initial"""
    def __init__(self, column: str, row_to_optimize: pd.DataFrame) -> None:
        self._column = column
        self._initial_value = row_to_optimize[column].iloc[0]

    def __call__(
        self, current_parameters: pd.DataFrame,
    ) -> pd.Series:
        return current_parameters[self._column] - self._initial_value

    def __repr__(self) -> str:
        return self.__class__.__name__
