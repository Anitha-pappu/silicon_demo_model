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
from abc import ABC, abstractmethod

import numpy.typing as npt
import pandas as pd

from modeling import types
from modeling.models.metrics_utils import evaluate_regression_metrics


class EvaluatesMetrics(ABC):

    def evaluate_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        Calculate a standard set of regression metrics for given data

        Args:
            data: data to calculate metrics
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            Mapping from metric name into metric value
        """
        target, prediction = self._get_target_and_prediction(data, **kwargs)
        return evaluate_regression_metrics(target, prediction)

    @abstractmethod
    def _get_target_and_prediction(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Tuple[npt.ArrayLike, types.Vector]:
        """
        Private method for inheritors
        that would produce actual and predicted target values
        for further metrics evaluation (this method is called by
        public method ``Self.evaluate_metrics``).

        Args:
            data: data to calculate metrics
            **kwargs: additional keyword arguments that
             are required for method implementation

        Returns:
            A tuple with actual and predicted target values
        """
