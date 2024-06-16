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
This submodule stores information about all model protocols that are used in reporting.
We are using structural subtyping to show interfaces of used models.
"""

import typing as tp

import pandas as pd
from typing_extensions import Self

from modeling import types

from ._estimator import Estimator  # noqa: WPS436


@tp.runtime_checkable
class SupportsModel(tp.Protocol):
    """Any models inherited from ``modeling.ModelBase`` implement this protocol"""
    def __init__(self, features_in: tp.Iterable[str], target: str) -> None:
        """Creates an instance of the model"""

    def predict(self, data: pd.DataFrame, **kwargs: tp.Any) -> types.Vector:
        """
        A public method for model prediction.

        Args:
            data: DataFrame to make a prediction
            **kwargs: Additional keyword arguments that
             might be passed for model prediction

        Returns:
            A Series or ndarray of model prediction
        """

    def fit(self: Self, data: pd.DataFrame, **kwargs: tp.Any) -> Self:
        """
        A public method to train model with data.

        Args:
            data: DataFrame to train model on
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Returns:
            A trained instance of BaseModel class
        """

    @property
    def features_in(self) -> tp.List[str]:
        """
        A property that contains columns that are required
        to be in the input dataset for `ModelBase` `.fit` or `.predict` methods.

        Returns:
            List of column names
        """

    @property
    def features_out(self) -> tp.List[str]:
        """
        A property containing the names of the features, produced as the
        result of transformations of the input dataset
        inside `.fit` or `.predict` methods.

        If no dataset transformations are applied, then ``.features_out`` property
        returns the same set of features as ``.features_in``.

        Returns:
            List of feature names
        """

    @property
    def target(self) -> str:
        """
        A property that contains model target column name.

        Returns:
            Column name
        """

    def get_feature_importance(
        self, data: pd.DataFrame, **kwargs: tp.Any,
    ) -> tp.Dict[str, float]:
        """
        A method for getting feature importance from model.

        Args:
            data: DataFrame to build feature importance
            **kwargs: Additional keyword arguments
             that might be passed for model training

        Note:
            This method returns mapping from ``features_out`` (feature set
            that is produced after all transformations step)
            to feature importance. This is expected because
            most of the feature importance extraction technics
            utilize estimators and return importance
            for feature set used by estimator.

        Returns:
            Dict with ``features_out`` as a keys and
            feature importances as a values
        """


@tp.runtime_checkable
class ContainsEstimator(tp.Protocol):
    """
    This protocol describes a model that can return an estimator-compatible object
    """
    @property
    def estimator(self) -> Estimator:
        """
        Returns sklearn estimator.

        Note:
            There are no assumptions about the input/output structure of the dataset
            that estimator expects. It's the user's responsibility to check input's
            and output's structure.
        """
