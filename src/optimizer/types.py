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
Holds custom types used throughout the code.
"""

import typing as tp

import numpy as np
import pandas as pd

Vector = tp.Union[np.ndarray[tp.Any, tp.Any], pd.Series]  # pylint: disable=E1136
Matrix = tp.Union[np.ndarray[tp.Any, tp.Any], np.matrix[tp.Any, tp.Any], pd.DataFrame]  # pylint: disable=E1136
MAXIMIZE: tp.Literal["maximize"] = "maximize"
MINIMIZE: tp.Literal["minimize"] = "minimize"
Sense = [MAXIMIZE, MINIMIZE]
TSense = tp.Literal["minimize", "maximize"]
TColumn = tp.Union[str, int]
TBounds = tp.Tuple[float, float]
TBoundsList = tp.List[TBounds]
TReal = tp.Union[float, int]
AllowedRealTypes = (float, int)
TCategorical = tp.Union[TReal, str, bool]


@tp.runtime_checkable
class ReducesMatrixToSeries(tp.Protocol):
    def __call__(self, parameters: Matrix, **kwargs: tp.Any) -> Vector:  # pylint: disable=W0613
        """Reduces input matrix to a vector"""


@tp.runtime_checkable
class MapsMatrixToMatrix(tp.Protocol):
    def __call__(self, parameters: Matrix, **kwargs: tp.Any) -> Matrix:  # pylint: disable=W0613
        """Maps input matrix to another matrix"""


@tp.runtime_checkable
class Predictor(tp.Protocol):
    """
    Protocol type class to define any class with a predict method.
    """

    def predict(self, parameters: Matrix, **kwargs: tp.Any) -> Vector:  # pylint: disable=W0613
        """Returns vector of predictions based on parameters"""


@tp.runtime_checkable
class StoresPenaltyValues(tp.Protocol):
    calculated_penalty: tp.Optional[Vector]


@tp.runtime_checkable
class HasName(tp.Protocol):
    @property
    def name(self) -> tp.Optional[str]:
        pass


@tp.runtime_checkable
class PenaltyCompatible(
    StoresPenaltyValues, HasName, ReducesMatrixToSeries, tp.Protocol,
):
    """
    User defined callable that has a name and stores penalty evaluation results.
    """
