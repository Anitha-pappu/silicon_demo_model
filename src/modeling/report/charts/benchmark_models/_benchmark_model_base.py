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


# todo: Benchmark models should not belong within reporting chart, but in their own
#  place in reporting
#  Then the charts should import them and use them where needed

import typing as tp
import warnings
from abc import ABC

import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from modeling.models.model_base import EvaluatesMetrics, ModelBase
from modeling.types import Vector


class BenchmarkModelBase(
    ModelBase,
    EvaluatesMetrics,
    ABC,
):
    """ Abstract class for benchmark models.

    Benchmark models are unsophisticated models intended to provide a benchmark for
    model performance when modeling.
    """

    def __init__(
        self,
        target: str,
        timestamp: str,
    ) -> None:
        super().__init__(features_in=[timestamp, target], target=target)
        self._timestamp = timestamp

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> Vector:
        """
        A public method for model prediction.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that might be passed for model
             prediction

        Returns:
            A Series or ndarray of model prediction
        """
        prediction = super().predict(data, **kwargs)
        class_name = self.__class__.__name__
        _warn_if_nans_are_present(
            data[self.features_in],
            message=(
                "Nulls detected in relevant data when predicting using an instance of"
                f"`{class_name}`. The prediction might be inaccurate."
            ),
        )
        return prediction

    def fit(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> Self:
        """
        A public method to train model with data.

        Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Returns:
            A trained instance of BaseModel class
        """
        self._fit(data, **kwargs)
        class_name = self.__class__.__name__
        _warn_if_nans_are_present(
            data[self.features_in],
            message=(
                "Nulls detected in relevant data when fitting an instance of"
                f"`{class_name}`."
            ),
        )
        return self

    @property
    def timestamp(self) -> str:
        """
        Returns the name of the timestamp column.

        Allows the user to know which of the feature has the "special" role of being
        the timestamp.
        """
        return self._timestamp

    def _get_target_and_prediction(
        self,
        data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> tp.Tuple[npt.ArrayLike, Vector]:
        """
        Pull a tuple of actual and predicted target values.

        Args:
            data: data to calculate metrics for
            **kwargs: additional keyword arguments to pass to
             self.predict() call

        Returns:
            A tuple containing actual and predicted target values
        """
        target = data[self._target].to_numpy()
        prediction = self.predict(data, **kwargs)
        return target, prediction


def _warn_if_nans_are_present(data: pd.DataFrame, message: str) -> None:
    """ Raises a warning if NaNs are present.

    Notes:
        The original use intended for this function is to warn of NaNs in
        ``data[self._feature_in]`` in the benchmark model
        Uses the provided ``message`` as a warning message
    """
    if data.isnull().any().any():
        warnings.warn(message)
