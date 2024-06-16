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
A protocol for objects to be cross-validatable.
"""
import typing as tp

import pandas as pd
from typing_extensions import Self

from ._evaluates_metrics import SupportsEvaluateMetrics  # noqa: WPS436


class TCrossValidatableModel(SupportsEvaluateMetrics, tp.Protocol):
    """
    A protocol that an object needs to adhere to be cross-validated.
    """

    def fit(self, data: pd.DataFrame, **kwargs: tp.Any) -> Self:
        """
        A public method to train model with data.

        Args:
            data: DataFrame to train the model on.
            **kwargs: Additional kwargs to pass for model training.

        Returns:
            A trained instance of the model.
        """
