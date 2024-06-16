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
from __future__ import annotations

import typing as tp
from datetime import datetime

import pandas as pd

from optimizer import StatefulOptimizationProblem

TRowToOptimize = pd.DataFrame
TProblem = tp.TypeVar('TProblem', bound=StatefulOptimizationProblem)
TIndexDType = tp.Union[str, float, datetime, pd.Timestamp, pd.DatetimeTZDtype]


@tp.runtime_checkable
class Predictor(tp.Protocol):
    """
    Protocol type class to define any class with a predict method.
    """

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> pd.Series:
        """Predicts target based on provided data"""


class ObjectiveFunction(tp.Protocol):
    # this is a protocol introduced in optimizer; and `parameters`
    # argument name comes from there; hence the noqa
    def __call__(self, parameters: pd.DataFrame) -> pd.Series:  # noqa: WPS110
        """
        Evaluates vector of objectives based on provided matrix of parameters
        with shape (parameters variations, initial columns)
        and returns a Series of objectives with shape (parameter variations,)
        """


@tp.runtime_checkable
class TagDictLike(tp.Protocol):
    def to_frame(self) -> pd.DataFrame:
        """Returns underlying dataset with tags metadata"""
